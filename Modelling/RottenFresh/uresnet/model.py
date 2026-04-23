"""UResNet model definition."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Basic residual block.

    args:
        in_ch: Number of input channels.
        out_ch: Number of output channels.
        stride: Stride used in the first convolution.

    returns:
        A residual feature extraction block with an identity or projection
        shortcut connection.
    """

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        """Apply residual learning to an input tensor.

        args:
            x: Input feature map.

        returns:
            Output feature map after the residual block.
        """
        identity = self.shortcut(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + identity)

        return x


class DoubleConv(nn.Module):
    """Double convolution block.

    args:
        in_ch: Number of input channels.
        out_ch: Number of output channels.

    returns:
        A two-layer convolutional feature extraction block.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        """Apply the double convolution block.

        args:
            x: Input feature map.

        returns:
            Output feature map after two convolutions.
        """
        return self.net(x)


class UBranch(nn.Module):
    """UNet branch.

    returns:
        A hierarchical convolutional branch that extracts spatial
        multi-scale features.
    """

    def __init__(self):
        super().__init__()

        self.enc1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128, 256)

    def forward(self, x):
        """Extract features using the U-Net-style branch.

        args:
            x: Input image tensor.

        returns:
            Deep feature map from the U-branch.
        """
        x = self.enc1(x)
        x = self.enc2(self.pool1(x))
        x = self.enc3(self.pool2(x))
        x = self.bottleneck(self.pool3(x))
        return x


class ResBranch(nn.Module):
    """ResNet branch.

    returns:
        A residual feature extraction branch for learning compact
        discriminative representations.
    """

    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer1 = ResidualBlock(64, 64)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)

    def forward(self, x):
        """Extract features using the ResNet-style branch.

        args:
            x: Input image tensor.

        returns:
            Deep feature map from the residual branch.
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class UResNetClassifier(nn.Module):
    """Final classifier.

    args:
        num_classes: Number of output classes.

    returns:
        A classifier that fuses U-branch and Res-branch features before
        producing class logits.
    """

    def __init__(self, num_classes):
        super().__init__()

        self.res = ResBranch()
        self.u = UBranch()

        self.fuse = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        """Run a forward pass through the UResNet classifier.

        args:
            x: Input image tensor.

        returns:
            Class logits for each input sample.
        """
        r = self.res(x)
        u = self.u(x)

        if u.shape[-2:] != r.shape[-2:]:
            u = F.interpolate(u, size=r.shape[-2:])

        x = torch.cat([r, u], dim=1)
        x = self.fuse(x)

        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)
