"""
This module contains the implementation of three different neural network models:
- ResNet-18
- ViT (pretrained)
"""

import torch.nn as nn
from transformers import ViTModel


class BasicBlock(nn.Module):
    """
    Basic Block for ResNet18.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        """
        Initializes the BasicBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride value for the convolutional layer.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        # Add a 1x1 convolutional layer to identity
        self.identity_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=stride, bias=False
        )

    def forward(self, x):
        """
        Forward pass of the BasicBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply the identity_conv layer to identity
        identity = self.identity_conv(identity)

        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    """
    ResNet18 model architecture.
    """

    def __init__(self, input_channels=3, num_classes=10):
        """
        Initializes the ResNet18 model.

        Args:
            num_classes (int): Number of classes in the dataset.
        """
        super(ResNet18, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.name = "ResNet18"

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        Helper function to create a layer of BasicBlocks.

        Args:
            block (nn.Module): BasicBlock or BottleneckBlock.
            out_channels (int): Number of output channels.
            num_blocks (int): Number of blocks in the layer.
            stride (int): Stride value for the first block in the layer.

        Returns:
            nn.Sequential: A sequential module of BasicBlocks.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the neural network model.

        Args:
            x (torch.Tensor): Input tensor to the model.

        Returns:
            torch.Tensor: Output tensor from the model.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def get_last_linear_layer(self):
        """
        Returns the last linear layer of the model.
        """
        return self.fc


class ViT(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(ViT, self).__init__()
        self.base = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.final = nn.Linear(self.base.config.hidden_size, num_classes)
        self.num_classes = num_classes
        self.relu = nn.ReLU()

    def forward(self, pixel_values):
        outputs = self.base(pixel_values=pixel_values)
        logits = self.final(outputs.last_hidden_state[:, 0])

        return logits

    def get_last_linear_layer(self):
        """
        Returns the last linear layer of the model.
        """
        return self.final
