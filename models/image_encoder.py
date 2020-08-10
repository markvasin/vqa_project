import torch
import torchvision
from torch import nn


class ResNet101ImageEncoder(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config
        model = torchvision.models.resnet101(pretrained=config.get("pretrained", True))
        modules = list(model.children())[:-3]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        # Bx3x224x224 -> Bx1024x14x14
        out = self.model(x)
        return out


class ImageBertEncoder(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config
        self.projection = nn.Conv2d(config.image_hidden_size, config.hidden_size, 1)

    def forward(self, x):
        # Bx2044x7x7 -> 2044x49
        out = self.projection(x)
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out
