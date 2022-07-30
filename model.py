import torch
import torch.nn as nn
from torch.nn.functional import adaptive_avg_pool1d


class ConvBNACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, 1, groups, False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class MicroBlock(nn.Module):
    def __init__(self, nh):
        super().__init__()
        self.conv1 = ConvBNACT(nh, nh, 1)
        self.conv2 = ConvBNACT(nh, nh, 3, 1, 1, nh)

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.conv2(x)
        return x


class MicroStage(nn.Sequential):
    def __init__(self, depth, nh):
        super().__init__(*[MicroBlock(nh) for _ in range(depth)])


class MLP(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(0.5)
        )


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fn(x)


class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.residual1 = Residual(MLP(input_dim, hidden_dim))
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.conv = nn.Conv2d(
            input_dim, input_dim, kernel_size=3, padding=1, groups=input_dim, bias=False)
        self.layer_norm3 = nn.LayerNorm(input_dim)
        self.residual2 = Residual(MLP(input_dim, hidden_dim))

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = self.layer_norm1(x)
        x = self.residual1(x)
        x = self.layer_norm2(x)
        x = x.permute(0, 3, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 3, 2, 1)
        x = self.layer_norm3(x)
        x = self.residual2(x)
        x = x.permute(0, 3, 2, 1)
        return x


class MLPStage(nn.Sequential):
    def __init__(self, depth, input_dim, hidden_dim):
        super().__init__(*[MLPBlock(input_dim, hidden_dim)
                           for _ in range(depth)])


class Tokenizer(nn.Sequential):
    def __init__(self, in_channels: int, hidden_dim: int, out_dim: int):
        super().__init__(
            ConvBNACT(in_channels, hidden_dim // 2, 3, 2, 1),
            ConvBNACT(hidden_dim // 2, hidden_dim // 2, 3, 1, 1),
            ConvBNACT(hidden_dim // 2, out_dim, 3, 1, 1),
            nn.MaxPool2d(3, 2, 1)
        )


class MicroMLPNet(nn.Module):
    def __init__(self, in_channels=3, nh=64, depth=2, nclass=60, img_height=32):
        super().__init__()
        """
        nh512可以
        """
        self.embed = Tokenizer(in_channels, nh, nh)
        self.micro_stages = MicroStage(depth, nh)
        self.mlp_stages = MLPStage(depth, nh, nh)
        self.flatten = nn.Flatten(1, 2)
        self.dropout = nn.Dropout(0.5)
        linear_in = nh * int(img_height//4)
        self.fc = nn.Linear(linear_in, nclass)

    def forward(self, x):
        x_shape = x.size()
        x = self.embed(x)
        x = self.micro_stages(x)
        x = self.mlp_stages(x)
        x = self.flatten(x)
        x = adaptive_avg_pool1d(x, int(x_shape[3]/4))
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    import time
    x = torch.randn(1, 3, 32, 128)
    model = MicroMLPNet(nh=32, depth=2, nclass=62, img_height=32)
    t0 = time.time()
    out = model(x)
    t1 = time.time()
    print(out.shape, (t1-t0)*1000)
    #torch.save(model, 'test.pth')
    #from torchsummaryX import summary
    #summary(model, x)
