import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=4):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.norm = LayerNorm2d(dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        identity = x
        x = self.norm(x)

        B, C, H, W = x.shape
        ws = self.window_size

        qkv = self.qkv(x)
        qkv = qkv.view(B, 3, self.num_heads, C // self.num_heads, H // ws, ws, W // ws, ws)
        qkv = qkv.permute(1, 0, 4, 6, 2, 5, 7, 3).contiguous()
        qkv = qkv.view(3, -1, self.num_heads, ws * ws, C // self.num_heads)

        q, k, v = qkv[0], qkv[1], qkv[2]

        with torch.amp.autocast('cuda', enabled=False):
            q_f32 = q.float()
            k_f32 = k.float()
            v_f32 = v.float()

            attn = (q_f32 @ k_f32.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)

            out = attn @ v_f32
            out = out.to(identity.dtype)

        out = out.view(B, H // ws, W // ws, self.num_heads, ws, ws, C // self.num_heads)
        out = out.permute(0, 3, 6, 1, 4, 2, 5).reshape(B, C, H, W)

        return identity + self.proj(out)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class UNetResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=True),
            SimpleGate(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            ChannelAttention(channels)
        )

    def forward(self, x):
        return x + self.block(self.norm(x))


class UNetDenoiser(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            UNetResBlock(64)
        )
        self.down1 = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(256, 128, kernel_size=1)
        )

        self.enc2 = nn.Sequential(
            UNetResBlock(128),
            UNetResBlock(128)
        )
        self.down2 = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(512, 256, kernel_size=1)
        )

        self.bottleneck = nn.Sequential(
            UNetResBlock(256),
            WindowAttention(256, window_size=8, num_heads=8),
            UNetResBlock(256)
        )

        self.up1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1),
            nn.PixelShuffle(2)
        )

        self.dec2_conv = nn.Conv2d(256, 128, kernel_size=1)
        self.dec2 = nn.Sequential(
            UNetResBlock(128),
            UNetResBlock(128)
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1),
            nn.PixelShuffle(2)
        )

        self.dec1_conv = nn.Conv2d(128, 64, kernel_size=1)
        self.dec1 = nn.Sequential(
            UNetResBlock(64),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        identity = x

        s1 = self.enc1(x)
        x = self.down1(s1)

        s2 = self.enc2(x)
        x = self.down2(s2)

        x = self.bottleneck(x)

        x = self.up1(x)
        x = torch.cat([x, s2], dim=1)
        x = self.dec2_conv(x)
        x = self.dec2(x)

        x = self.up2(x)
        x = torch.cat([x, s1], dim=1)
        x = self.dec1_conv(x)
        x = self.dec1(x)

        return identity - x
