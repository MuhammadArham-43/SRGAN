import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_bn: bool = True,
        is_discriminator: bool = False,
        use_act: bool = False,
        **kwargs
    ) -> None:
        super().__init__()
        self.use_act = use_act
        self.cnn = nn.Conv2d(in_channels, out_channels,
                             **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True) if is_discriminator else nn.PReLU(
            num_parameters=out_channels)

    def forward(self, x):
        x = self.cnn(x)
        x = self.bn(x)
        x = self.act(x) if self.use_act else x
        return x
