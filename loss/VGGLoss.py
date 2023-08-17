import torch.nn as nn
from torchvision.models.vgg import vgg19, VGG19_Weights


class VGGLoss(nn.Module):
    def __init__(
        self,
        device='cpu'
    ) -> None:
        super().__init__()

        self.vgg = vgg19(
            weights=VGG19_Weights.DEFAULT
        ).features[:36].eval().to(device)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, inp, tgt):
        inp_features = self.vgg(inp)
        tgt_features = self.vgg(tgt)
        return self.loss(tgt_features, inp_features)


if __name__ == "__main__":
    import torch
    inp = torch.randn(3, 96, 96)
    tgt = torch.randn(3, 96, 96)

    l = VGGLoss()
    print(l(inp, tgt))
