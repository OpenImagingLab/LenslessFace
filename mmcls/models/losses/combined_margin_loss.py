# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn
import numpy as np
import math
import torch
from ..builder import LOSSES
import collections
from typing import Callable
from torch import distributed
from torch.nn.functional import linear, normalize
from .cross_entropy_loss import CrossEntropyLoss
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
#import torch.fft.fftn
# ArcFace: m1=1.0, m3=0.0, m2 is the margin
# CosFace: m3 > 0, (m1 and m2 is useless)
@LOSSES.register_module()
class CombinedMarginLoss(torch.nn.Module):
    def __init__(self, s, m1, m2, m3,
                 interclass_filtering_threshold=0,
                 ce_cfg=None):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold

        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False

        # For cross entropy loss
        self.cross_entropy_loss = CrossEntropyLoss(**ce_cfg)

    def forward(self, logits, labels, **kwargs):
        index_positive = torch.where(labels != -1)[0]

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)]

        if self.m1 == 1.0 and self.m3 == 0.0:
            sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
            cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
            if self.easy_margin:
                final_target_logit = torch.where(
                    target_logit > 0, cos_theta_m, target_logit)
            else:
                final_target_logit = torch.where(
                    target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s

        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        else:
            raise

        loss = self.cross_entropy_loss(logits, labels, **kwargs)
        return loss


class DistCrossEntropyFunc(torch.autograd.Function):
    """
    CrossEntropy loss is calculated in parallel, allreduce denominator into single gpu and calculate softmax.
    Implemented of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, label: torch.Tensor):
        """ """
        batch_size = logits.size(0)
        # for numerical stability
        max_logits, _ = torch.max(logits, dim=1, keepdim=True)
        # local to global
        distributed.all_reduce(max_logits, distributed.ReduceOp.MAX)
        logits.sub_(max_logits)
        logits.exp_()
        sum_logits_exp = torch.sum(logits, dim=1, keepdim=True)
        # local to global
        distributed.all_reduce(sum_logits_exp, distributed.ReduceOp.SUM)
        logits.div_(sum_logits_exp)
        index = torch.where(label != -1)[0]
        # loss
        loss = torch.zeros(batch_size, 1, device=logits.device)
        loss[index] = logits[index].gather(1, label[index])
        distributed.all_reduce(loss, distributed.ReduceOp.SUM)
        ctx.save_for_backward(index, logits, label)
        return loss.clamp_min_(1e-30).log_().mean() * (-1)

    @staticmethod
    def backward(ctx, loss_gradient):
        """
        Args:
            loss_grad (torch.Tensor): gradient backward by last layer
        Returns:
            gradients for each input in forward function
            `None` gradients for one-hot label
        """
        (
            index,
            logits,
            label,
        ) = ctx.saved_tensors
        batch_size = logits.size(0)
        one_hot = torch.zeros(
            size=[index.size(0), logits.size(1)], device=logits.device
        )
        one_hot.scatter_(1, label[index], 1)
        logits[index] -= one_hot
        logits.div_(batch_size)
        return logits * loss_gradient.item(), None


class DistCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(DistCrossEntropy, self).__init__()

    def forward(self, logit_part, label_part):
        return DistCrossEntropyFunc.apply(logit_part, label_part)


class PartialFC(torch.nn.Module):
    """
    https://arxiv.org/abs/2203.15565
    A distributed sparsely updating variant of the FC layer, named Partial FC (PFC).

    When sample rate less than 1, in each iteration, positive class centers and a random subset of
    negative class centers are selected to compute the margin-based softmax loss, all class
    centers are still maintained throughout the whole training process, but only a subset is
    selected and updated in each iteration.

    .. note::
        When sample rate equal to 1, Partial FC is equal to model parallelism(default sample rate is 1).

    Example:
    --------
    >>> module_pfc = PartialFC(embedding_size=512, num_classes=8000000, sample_rate=0.2)
    >>> for img, labels in data_loader:
    >>>     embeddings = net(img)
    >>>     loss = module_pfc(embeddings, labels, optimizer)
    >>>     loss.backward()
    >>>     optimizer.step()
    """
    _version = 1

    def __init__(
            self,
            margin_loss: Callable,
            embedding_size: int,
            num_classes: int,
            sample_rate: float = 1.0,
            fp16: bool = False,
    ):
        """
        Paramenters:
        -----------
        embedding_size: int
            The dimension of embedding, required
        num_classes: int
            Total number of classes, required
        sample_rate: float
            The rate of negative centers participating in the calculation, default is 1.0.
        """
        super(PartialFC, self).__init__()
        assert (
            distributed.is_initialized()
        ), "must initialize distributed before create this"
        self.rank = distributed.get_rank()
        self.world_size = distributed.get_world_size()

        self.dist_cross_entropy = DistCrossEntropy()
        self.embedding_size = embedding_size
        self.sample_rate: float = sample_rate
        self.fp16 = fp16
        self.num_local: int = num_classes // self.world_size + int(
            self.rank < num_classes % self.world_size
        )
        self.class_start: int = num_classes // self.world_size * self.rank + min(
            self.rank, num_classes % self.world_size
        )
        self.num_sample: int = int(self.sample_rate * self.num_local)
        self.last_batch_size: int = 0
        self.weight: torch.Tensor
        self.weight_mom: torch.Tensor
        self.weight_activated: torch.nn.Parameter
        self.weight_activated_mom: torch.Tensor
        self.is_updated: bool = True
        self.init_weight_update: bool = True

        if self.sample_rate < 1:
            self.register_buffer("weight",
                                 tensor=torch.normal(0, 0.01, (self.num_local, embedding_size)))
            self.register_buffer("weight_mom",
                                 tensor=torch.zeros_like(self.weight))
            self.register_parameter("weight_activated",
                                    param=torch.nn.Parameter(torch.empty(0, 0)))
            self.register_buffer("weight_activated_mom",
                                 tensor=torch.empty(0, 0))
            self.register_buffer("weight_index",
                                 tensor=torch.empty(0, 0))
        else:
            self.weight_activated = torch.nn.Parameter(torch.normal(0, 0.01, (self.num_local, embedding_size)))

        # margin_loss
        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise

    @torch.no_grad()
    def sample(self,
               labels: torch.Tensor,
               index_positive: torch.Tensor,
               optimizer: torch.optim.Optimizer):
        """
        This functions will change the value of labels

        Parameters:
        -----------
        labels: torch.Tensor
            pass
        index_positive: torch.Tensor
            pass
        optimizer: torch.optim.Optimizer
            pass
        """
        positive = torch.unique(labels[index_positive], sorted=True).cuda()
        if self.num_sample - positive.size(0) >= 0:
            perm = torch.rand(size=[self.num_local]).cuda()
            perm[positive] = 2.0
            index = torch.topk(perm, k=self.num_sample)[1].cuda()
            index = index.sort()[0].cuda()
        else:
            index = positive
        self.weight_index = index

        labels[index_positive] = torch.searchsorted(index, labels[index_positive])

        self.weight_activated = torch.nn.Parameter(self.weight[self.weight_index])
        self.weight_activated_mom = self.weight_mom[self.weight_index]

        if isinstance(optimizer, torch.optim.SGD):
            # TODO the params of partial fc must be last in the params list
            optimizer.state.pop(optimizer.param_groups[-1]["params"][0], None)
            optimizer.param_groups[-1]["params"][0] = self.weight_activated
            optimizer.state[self.weight_activated][
                "momentum_buffer"
            ] = self.weight_activated_mom
        else:
            raise

    @torch.no_grad()
    def update(self):
        """ partial weight to global
        """
        if self.init_weight_update:
            self.init_weight_update = False
            return

        if self.sample_rate < 1:
            self.weight[self.weight_index] = self.weight_activated
            self.weight_mom[self.weight_index] = self.weight_activated_mom

    def forward(
            self,
            local_embeddings: torch.Tensor,
            local_labels: torch.Tensor,
            optimizer: torch.optim.Optimizer,
    ):
        """
        Parameters:
        ----------
        local_embeddings: torch.Tensor
            feature embeddings on each GPU(Rank).
        local_labels: torch.Tensor
            labels on each GPU(Rank).

        Returns:
        -------
        loss: torch.Tensor
            pass
        """
        local_labels.squeeze_()
        local_labels = local_labels.long()
        self.update()

        batch_size = local_embeddings.size(0)
        if self.last_batch_size == 0:
            self.last_batch_size = batch_size
        assert self.last_batch_size == batch_size, (
            "last batch size do not equal current batch size: {} vs {}".format(
                self.last_batch_size, batch_size))

        _gather_embeddings = [
            torch.zeros((batch_size, self.embedding_size)).cuda()
            for _ in range(self.world_size)
        ]
        _gather_labels = [
            torch.zeros(batch_size).long().cuda() for _ in range(self.world_size)
        ]
        _list_embeddings = AllGather(local_embeddings, *_gather_embeddings)
        distributed.all_gather(_gather_labels, local_labels)

        embeddings = torch.cat(_list_embeddings)
        labels = torch.cat(_gather_labels)

        labels = labels.view(-1, 1)
        index_positive = (self.class_start <= labels) & (
                labels < self.class_start + self.num_local
        )
        labels[~index_positive] = -1
        labels[index_positive] -= self.class_start

        if self.sample_rate < 1:
            self.sample(labels, index_positive, optimizer)

        with torch.cuda.amp.autocast(self.fp16):
            norm_embeddings = normalize(embeddings)
            norm_weight_activated = normalize(self.weight_activated)
            logits = linear(norm_embeddings, norm_weight_activated)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)

        logits = self.margin_softmax(logits, labels)
        loss = self.dist_cross_entropy(logits, labels)
        return loss

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = collections.OrderedDict()
            destination._metadata = collections.OrderedDict()

        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + ".", keep_vars=keep_vars)
        if self.sample_rate < 1:
            destination["weight"] = self.weight.detach()
        else:
            destination["weight"] = self.weight_activated.data.detach()
        return destination

    def load_state_dict(self, state_dict, strict: bool = True):
        if self.sample_rate < 1:
            self.weight = state_dict["weight"].to(self.weight.device)
            self.weight_mom.zero_()
            self.weight_activated.data.zero_()
            self.weight_activated_mom.zero_()
            self.weight_index.zero_()
        else:
            self.weight_activated.data = state_dict["weight"].to(self.weight_activated.data.device)


class AllGatherFunc(torch.autograd.Function):
    """AllGather op with gradient backward"""

    @staticmethod
    def forward(ctx, tensor, *gather_list):
        gather_list = list(gather_list)
        distributed.all_gather(gather_list, tensor)
        return tuple(gather_list)

    @staticmethod
    def backward(ctx, *grads):
        grad_list = list(grads)
        rank = distributed.get_rank()
        grad_out = grad_list[rank]

        dist_ops = [
            distributed.reduce(grad_out, rank, distributed.ReduceOp.SUM, async_op=True)
            if i == rank
            else distributed.reduce(
                grad_list[i], i, distributed.ReduceOp.SUM, async_op=True
            )
            for i in range(distributed.get_world_size())
        ]
        for _op in dist_ops:
            _op.wait()

        grad_out *= len(grad_list)  # cooperate with distributed loss function
        return (grad_out, *[None for _ in range(len(grad_list))])


AllGather = AllGatherFunc.apply

@LOSSES.register_module()
class ArcMargin(nn.Module):
    def __init__(self, in_features=128, out_features=200, s=32.0, m=0.50, loss_weight=1.0,easy_margin=False):
        super(ArcMargin, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.loss_weight = loss_weight
        nn.init.xavier_uniform_(self.weight)
        # init.kaiming_uniform_()
        # self.weight.data.normal_(std=0.001)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label, **kwargs):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
	
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, label)
        return self.loss_weight * loss
        #return output

@LOSSES.register_module()
class TV_loss(nn.Module):
    def __init__(self,loss_weight = 1.0):
        super(TV_loss,self).__init__()
        self.weight =-1* loss_weight
    def forward(self,k):
        h_k = k.size()[-2]
        w_k = k.size()[-1]
        count_h = (k.size()[-2]-1)*k.size()[-1]
        count_w = k.size()[-2]*(k.size()[-1]-1)
        h_tv = torch.pow((k[:,:,1:,:]-k[:,:,:h_k-1,:]),2).sum()
        w_tv = torch.pow((k[:,:,:,1:]-k[:,:,:,:w_k-1]),2).sum()
        

        return self.weight*2*(h_tv/count_h+w_tv/count_w)
    

@LOSSES.register_module()
class bin_loss(nn.Module):
    def __init__(self,loss_weight = 1.0):
        super(bin_loss,self).__init__()
        self.weight = loss_weight
    def forward(self,k):
        h_k = k.size()[-2]
        w_k = k.size()[-1]
        loss = 0
        for i in range(h_k):
            for j in range(w_k):
                sq1 = torch.pow(k[:,:,i,j],2)
                sq2 = torch.pow((k[:,:,i,j]-1),2)
                loss = loss+sq1+sq2
        loss = loss/(h_k*w_k)
        return torch.mean(self.weight*loss)

@LOSSES.register_module()
class trans_loss(nn.Module):
    def __init__(self,loss_weight = 1.0):
        super(trans_loss,self).__init__()
        self.weight = loss_weight
    def forward(self,k):
        h_k = k.size()[-2]
        w_k = k.size()[-1]
        loss = torch.sum(k)/(h_k*w_k)
        return self.weight*loss


@LOSSES.register_module()
class inv_loss(nn.Module):
    def __init__(self,loss_weight = 1.0):
        super(inv_loss,self).__init__()
        self.weight =1* loss_weight
    def forward(self,k):
        device = k.device
        k = k.cpu().detach().numpy()   
        
        kernel_fft = np.fft.fftn(k,axes=(-2,-1))
        kernel_fft =torch.abs(torch.from_numpy(kernel_fft))
        kernel_fft = kernel_fft.to(device)
        
        loss = torch.norm(kernel_fft,p=1,dim=(-1,-2))

        return torch.mean(self.weight*loss)




def hxx(x, y):
    size = x.shape[-1]
    px = np.histogram(x, 256, (0, 255))[0] / size
    py = np.histogram(y, 256, (0, 255))[0] / size
    hx = - np.sum(px * np.log(px + 1e-8))
    hy = - np.sum(py * np.log(py + 1e-8))

    hxy = np.histogram2d(x, y, 256, [[0, 255], [0, 255]])[0]
    hxy /= (1.0 * size)
    hxy = - np.sum(hxy * np.log(hxy + 1e-8))

    r = hx + hy - hxy
    return r

@LOSSES.register_module()
class entropy_loss(nn.Module):
    def __init__(self,loss_weight = 1.0):
        super(entropy_loss,self).__init__()
        self.weight = loss_weight
   
    def hxx(x, y):
        size = x.shape[-1]
        px = np.histogram(x, 256, (0, 255))[0] / size
        py = np.histogram(y, 256, (0, 255))[0] / size
        hx = - np.sum(px * np.log(px + 1e-8))
        hy = - np.sum(py * np.log(py + 1e-8))
 
        hxy = np.histogram2d(x, y, 256, [[0, 255], [0, 255]])[0]
        hxy /= (1.0 * size)
        hxy = - np.sum(hxy * np.log(hxy + 1e-8))
 
        r = hx + hy - hxy
        return r

    def forward(self,x0,img):
       # print(img.shape)
       # print(x0)
       # print(x0.shape)
        x0_reshape = torch.nn.functional.interpolate(x0,size=(112,96))
        img = img.cpu().detach().numpy()
        x0 = x0_reshape.cpu().detach().numpy()
        x0 = np.uint8(x0)
        img = np.uint8(img)
      #  print(x0.type)
       # print(img.type)
        loss = hxx(img,x0)
        loss = torch.from_numpy(loss)
        print(loss)
        return self.weight*loss
        
             
@LOSSES.register_module()
class hybridloss(nn.Module):
    def __init__(self,m1,m2,m3,m4,m5,out_features):
        super(hybridloss,self).__init__()
        self.weight1 = m1
        self.weight2 = m2
        self.weight3 = m3
        self.weight4 = m4
        self.weight5 = m5
        self.arcmargin = ArcMargin(out_features=out_features)
        self.tvloss = TV_loss()
        self.binloss = bin_loss()
      #  self.transloss = trans_loss()
        self.entropy_loss = entropy_loss()
        self.invloss = inv_loss()
    def forward(self,x, label, k,x0,img, **kwargs):
        loss_1 = self.weight1*self.arcmargin(x,label,**kwargs)
        print(loss_1)
        loss_2 = self.weight2*self.tvloss(k)
        print(loss_2)
        loss_3 = self.weight3*self.binloss(k)
        print(loss_3)
       # loss_4 = self.weight4*self.entropy_loss(img,x0)
       # print(loss_4)
        loss_5 = self.weight5*self.invloss(k)
        print(loss_5)
        loss = loss_1+loss_2+loss_3+loss_5
        print(loss)
        return loss
