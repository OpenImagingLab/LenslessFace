import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.conv import _pair


class BinarizeFunction(Function):
    @staticmethod
    def forward(ctx, x):
        y = (x + 1.0) / 2.0
        y = y.clamp(min=0, max=1)
        y = torch.round(y)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        grad_x = grad_y
        return grad_x


binarization = BinarizeFunction.apply

class WeightFunction(Function):
    @staticmethod
    def forward(ctx,x):
        b, c, w, h = x.shape
        weight_mean = torch.zeros([1, c, h, w])
        weight = torch.zeros([b, c, h, w])
        for i in range(b):
            weight_mean += x[i,:,:,:]
        weight_mean = torch.div(weight_mean,b)
        for i in range(b):
            weight[i,:,:,:] = weight_mean
        return weight
    @staticmethod
    def backward(ctx,grad_weight):
        grad_x = grad_weight
        return grad_weight

weightequal = WeightFunction.apply


class BinaryConv2d(nn.Conv2d):
    def _conv_forward(self, input, weight, bias=None):
        b, c, h, w = weight.shape
        with torch.no_grad():
            for i in range(1,b):
                weight[i,:,:,:] = weight[0,:,:,:]
        if self.padding_mode != 'zeros':
            return F.conv2d(
                F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                binarization(weight),
                binarization(self.bias) if self.bias is not None else self.bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups
            )
       # print(weight)
    #    print(binarization(weight))
        return F.conv2d(input,
                        binarization(weight),
                        binarization(self.bias) if self.bias is not None else self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


class BinaryConv2d_cx(nn.Conv2d):
    def _conv_forward(self, input, weight, bias):
 
        if self.padding_mode != 'zeros':
            return F.conv2d(
                F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                binarization(weight),
                binarization(bias) if bias is not None else bias,
                self.stride,
                _pair(0),
                self.dilation,
                1
            )

        conv_r = F.conv2d(input[:,0:1,:,:],
                          binarization(weight),
                          binarization(bias) if bias is not None else bias,
                          self.stride, self.padding, self.dilation, 1)
        conv_g = F.conv2d(input[:,1:2,:,:],   
                          binarization(weight),
                          binarization(bias) if bias is not None else bias,
                          self.stride, self.padding, self.dilation, 1)
        conv_b = F.conv2d(input[:,2:3,:,:],
                          binarization(weight),
                          binarization(bias) if bias is not None else bias,
                          self.stride, self.padding, self.dilation, 1)
        conv_result = torch.cat((conv_r,conv_g,conv_b),dim=1)
        # print(conv_result, weight)
        return conv_result


