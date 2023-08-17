import torch
import torch.nn as nn

from .convBlock import ConvBlock

from torchsummary import summary


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.block1 = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            use_bn=True,
            is_discriminator=False,
            use_act=True,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.block2 = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            use_bn=True,
            is_discriminator=False,
            use_act=False,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(x)
        return out + x


class UpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        scale_factor: int,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cnn = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * (2 ** scale_factor),
            kernel_size=3,
            stride=1,
            padding=1
        )

        # in_c * s ** 2 x H x W -> in_c x H * 2 x W * 2
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scale_factor)
        self.act = nn.PReLU(num_parameters=in_channels)

    def forward(self, x):
        return self.act(self.pixel_shuffle(self.cnn(x)))


class Generator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_channels: int = 64,
        num_blocks: int = 16,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.inital_cnn = ConvBlock(
            in_channels=in_channels,
            out_channels=num_channels,
            use_bn=False,
            is_discriminator=False,
            use_act=True,
            kernel_size=9,
            stride=1,
            padding=4,
        )

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(in_channels=num_channels) for _ in range(num_blocks)
        ])

        self.conv_block = ConvBlock(
            in_channels=num_channels,
            out_channels=num_channels,
            use_bn=True,
            is_discriminator=False,
            use_act=False,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.upsample_blocks = nn.ModuleList([
            UpsampleBlock(in_channels=num_channels, scale_factor=2) for i in range(2)
        ])

        self.out_block = nn.Conv2d(
            in_channels=num_channels,
            out_channels=in_channels,
            stride=1,
            kernel_size=9,
            padding=4
        )

    def forward(self, initial):
        initial = self.inital_cnn(initial)
        x = initial
        for block in self.residual_blocks:
            x = block(x)

        x = self.conv_block(x) + initial

        for block in self.upsample_blocks:
            x = block(x)

        x = self.out_block(x)

        return torch.tanh(x)


if __name__ == "__main__":
    inp = torch.randn(1, 3, 24, 24)
    gen = Generator(
        in_channels=3,
        num_channels=64,
        num_blocks=16
    )

    from discriminator import Discriminator
    # summary(gen, inp)
    z = gen(inp)
    print(z.shape)
    disc = Discriminator(in_channels=3)
    out = disc(z)
    print(out.shape, out)
