import torch
import torch.nn as nn

from .convBlock import ConvBlock

from torchsummary import summary

from typing import List


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        features: List[int] = [64, 64, 128, 128, 256, 256, 512, 512],
    ) -> None:
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        for idx, feature_dim in enumerate(features):
            self.conv_blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=feature_dim,
                    use_bn=False if idx == 0 else True,
                    is_discriminator=True,
                    use_act=True,
                    stride=1 + (idx % 2),
                    kernel_size=3,
                    padding=1
                )
            )
            in_channels = feature_dim

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)

        return self.classifier(x)


if __name__ == "__main__":
    disc = Discriminator(in_channels=3)
    inp = torch.randn(1, 3, 96, 96)
    summary(disc, inp)
