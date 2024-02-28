# Spacial transformer network
# https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3,
                               bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(126720, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 2)
        )
   
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)
        self.xs = xs
        self.theta = self.fc_loc(xs)
        self.theta = self.theta.view(-1, 2, 3)
        grid = F.affine_grid(self.theta, x.size(),align_corners=True)
        x = F.grid_sample(x, grid,align_corners=True)

        return x

if __name__  == "__main__":
    #unit test for STN
    x = torch.randn(1, 3, 164, 128)
    stn = STN()
    print(stn(x).shape)