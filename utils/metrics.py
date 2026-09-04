import torch
import math


def psnr(img1, img2, max_val=1.0):
    """
    Вычисляет PSNR между двумя изображениями.
    Аргументы:
        img1, img2: тензоры одинаковой формы, значения в диапазоне [0, max_val]
        max_val: максимальное значение пикселя (по умолчанию 1.0)
    Возвращает:
        PSNR в дБ (float)
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val / math.sqrt(mse))


def rmse(img1, img2):
    """
    Вычисляет RMSE между двумя изображениями.
    Аргументы:
        img1, img2: тензоры одинаковой формы
    Возвращает:
        RMSE (float)
    """
    mse = torch.mean((img1 - img2) ** 2)
    return torch.sqrt(mse)
