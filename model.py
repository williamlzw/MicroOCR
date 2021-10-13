import torch
import torch.nn as nn


class ConvBNACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups,
                              bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class MicroBlock(nn.Module):
    def __init__(self, nh, kernel_size):
        super().__init__()
        self.conv1 = ConvBNACT(nh, nh, kernel_size, groups=nh, padding=1)
        self.conv2 = ConvBNACT(nh, nh, 1)

    def forward(self, x):
        x = x + self.conv1(x)
        x = self.conv2(x)
        return x


class MicroNet(nn.Module):
    def __init__(self, nh=64, depth=2, nclass=60, img_width=120):
        super().__init__()
        self.conv = ConvBNACT(3, nh, 4, 4)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(MicroBlock(nh, 3))
        self.flatten = nn.Flatten(2)
        self.avg_pool = nn.AdaptiveAvgPool1d(int(img_width/4))
        self.fc = nn.Linear(nh, nclass)

    def forward(self, x):
        x = self.conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.flatten(x)
        x = self.avg_pool(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    import time
    x = torch.randn(1, 3, 32, 128)
    model = MicroNet(64)
    t0 = time.time()
    out = model(x)
    t1 = time.time()
    print(out.shape, (t1-t0)*1000)
    from torchsummary import summary
    summary(model, (3, 32, 128), device='cpu')
