import torch
import torchvision
from mmf.modules.layers import ConvNet, Flatten
from torch import nn


class ResNet101ImageEncoder(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config
        model = torchvision.models.resnet101(pretrained=True)
        modules = list(model.children())[:-3]
        self.resnet = nn.Sequential(*modules)
        self.projection = nn.Conv2d(config.image_hidden_size, config.hidden_size, 1)
        for p in self.resnet.parameters():
            p.requires_grad = False

    def forward(self, x):
        # Bx3x224x224 -> Bx1024x14x14
        out = self.resnet(x)
        out = self.projection(out)
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out


class ImageBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.projection = nn.Conv2d(config.image_hidden_size, config.hidden_size, 1)

    def forward(self, x):
        # Bx2044x7x7 -> 2044x49
        out = self.projection(x)
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out


class ImageClevrEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv1 = ConvNet(3, config.hidden_size, kernel_size=3)
        self.conv2 = ConvNet(config.hidden_size, config.hidden_size, kernel_size=3)
        self.conv3 = ConvNet(config.hidden_size, config.hidden_size, kernel_size=3)
        self.conv4 = ConvNet(config.hidden_size, config.hidden_size, kernel_size=3)

    def forward(self, image):
        out = self.conv1(image)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out
