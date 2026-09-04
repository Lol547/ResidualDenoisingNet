import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channel = 3

        window = self._create_window(self.window_size, self.sigma, self.channel)
        self.register_buffer('window', window)

    def _gaussian(self, window_size, sigma):
        gauss = torch.tensor(
            [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def _create_window(self, window_size, sigma, channel):
        _1D_window = self._gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
        return _2D_window.expand(channel, 1, window_size, window_size).contiguous().float()

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()

        if channel != self.channel:
            window = self._create_window(self.window_size, self.sigma, channel)
            self.register_buffer('window', window)
            self.channel = channel

        window_actual = self.window.to(device=img1.device, dtype=img1.dtype)

        mu1 = F.conv2d(img1, window_actual, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window_actual, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = torch.clamp(
            F.conv2d(img1 * img1, window_actual, padding=self.window_size // 2, groups=channel) - mu1_sq, min=0.0)
        sigma2_sq = torch.clamp(
            F.conv2d(img2 * img2, window_actual, padding=self.window_size // 2, groups=channel) - mu2_sq, min=0.0)
        sigma12 = F.conv2d(img1 * img2, window_actual, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        ssim_denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim_map = ssim_numerator / ssim_denominator

        return 1.0 - ssim_map.mean()


class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.vgg_submodel = nn.Sequential(*list(vgg.children())[:16]).to(device).eval()

        for param in self.vgg_submodel.parameters():
            param.requires_grad = False

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, pred, target):
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std

        f_pred = self.vgg_submodel(pred_norm)
        f_target = self.vgg_submodel(target_norm)

        return F.mse_loss(f_pred, f_target)
