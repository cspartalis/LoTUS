import torch
import torch.nn as nn
import torch.nn.functional as F

"""
This module contains the implementation of three different neural network models:
- ResNet-18
- All-CNN
- VGG-19
- ViT
"""

import torch.nn as nn
import torch.nn.functional as F


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

    def get_last_fc_layer(self):
        """
        Returns the last linear layer of the model.
        """
        return self.fc


class AllCNN(nn.Module):
    """
    All-CNN model architecture.
    """

    def __init__(self, input_channels=3, num_classes=10):
        """
        Initializes the AllCNN model.

        Args:
            num_classes (int): Number of classes in the dataset.
            input_channels (int): Number of input channels in the image.
        """
        super(AllCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            input_channels, 96, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv7 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv8 = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv9 = nn.Conv2d(
            192, num_classes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.name = "AllCNN"

    def forward(self, x):
        """
        Forward pass of the neural network model.

        Args:
            x (torch.Tensor): Input tensor to the model.

        Returns:
            torch.Tensor: Output tensor from the model.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class VGG19(nn.Module):
    """
    VGG-19 model architecture.
    """

    def __init__(self, input_channels=3, num_classes=10):
        """
        Initializes the VGG-19 model.

        Args:
            num_classes (int): Number of classes in the dataset.
            input_channels (int): Number of input channels in the image.
        """
        super(VGG19, self).__init__()
        self.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(
            512, 512, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn10 = nn.BatchNorm2d(512)
        self.conv11 = nn.Conv2d(
            512, 512, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(
            512, 512, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn12 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv13 = nn.Conv2d(
            512, 512, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn13 = nn.BatchNorm2d(512)
        self.conv14 = nn.Conv2d(
            512, 512, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn14 = nn.BatchNorm2d(512)
        self.conv15 = nn.Conv2d(
            512, 512, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn15 = nn.BatchNorm2d(512)
        self.conv16 = nn.Conv2d(
            512, 512, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn16 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        self.name = "VGG19"

    def forward(self, x):
        """
        Forward pass of the neural network model.

        Args:
            x (torch.Tensor): Input tensor to the model.

        Returns:
            torch.Tensor: Output tensor from the model.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool3(x)
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = self.pool4(x)
        x = F.relu(self.bn13(self.conv13(x)))
        x = F.relu(self.bn14(self.conv14(x)))
        x = F.relu(self.bn15(self.conv15(x)))
        x = F.relu(self.bn16(self.conv16(x)))
        x = self.pool5(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc3(x)
        return x

    def get_last_fc_layer(self):
        """
        Returns the last linear layer of the model.
        """
        return self.fc3


# ================================================================================
# ================================== V i T =======================================
# ================================================================================

# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.1
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
