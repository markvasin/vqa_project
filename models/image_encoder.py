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
        # Bx3x224x224 -> Bx2048x7x7
        out = self.model(x)
        return out
