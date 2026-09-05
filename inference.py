import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

from models import UNetDenoiser


class ImageDenoiser:
    def __init__(self, weights_path: str):
        """
        Инициализирует класс, определяет доступное устройство и загружает веса.

        :param weights_path: Абсолютный или относительный путь к файлу с весами модели (.pth).
        :raises FileNotFoundError: Если файл весов не найден по указанному пути.
        """
        print("Инициализация модуля очистки изображений...")

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Файл весов не найден: {weights_path}")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Устройство для вычислений: {self.device.type.upper()}")

        self.model = UNetDenoiser().to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
        self.model.eval()

        print("Нейросеть успешно загружена и готова к работе.")

    def process_image(self, image_path: str, tile_size: int = 800, tile_overlap: int = 64):
        """
        Обрабатывает указанное изображение через нейросеть для удаления шума.
        Автоматически решает проблемы с размерностью картинки (добавляет временный
        padding до кратности 32, необходимый для работы архитектуры UNet).

        :param image_path: Путь к исходному (зашумленному) изображению.
        :param tile_size: Максимальный размер кусочка (тайла).
        :param tile_overlap: Размер нахлеста между тайлами (скрывает швы при склейке).
        :return: Кортеж из двух объектов Pillow: (Оригинал, Очищенная_картинка).
        """
        print(f"Чтение файла: {image_path}...")

        img_pil = Image.open(image_path).convert('RGB')
        transform = transforms.ToTensor()
        img_tensor = transform(img_pil).unsqueeze(0)

        orig_h, orig_w = img_tensor.shape[2], img_tensor.shape[3]

        if tile_size is None or (orig_h <= tile_size and orig_w <= tile_size):
            img_tensor = img_tensor.to(self.device)

            pad_h = (32 - orig_h % 32) % 32
            pad_w = (32 - orig_w % 32) % 32
            img_padded = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h),
                               mode='reflect') if pad_h > 0 or pad_w > 0 else img_tensor

            with torch.no_grad():
                if self.device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        denoised_padded = self.model(img_padded)
                else:
                    denoised_padded = self.model(img_padded)
                denoised_padded = torch.clamp(denoised_padded, 0.0, 1.0)

            denoised_final = denoised_padded[:, :, :orig_h, :orig_w].cpu()

        else:
            stride = tile_size - tile_overlap

            output = torch.zeros_like(img_tensor)
            count = torch.zeros_like(img_tensor)

            for y in range(0, orig_h, stride):
                for x in range(0, orig_w, stride):
                    y1, y2 = y, min(y + tile_size, orig_h)
                    x1, x2 = x, min(x + tile_size, orig_w)

                    tile = img_tensor[:, :, y1:y2, x1:x2].to(self.device)

                    th, tw = tile.shape[2], tile.shape[3]
                    pad_h = (32 - th % 32) % 32
                    pad_w = (32 - tw % 32) % 32
                    tile_padded = torch.nn.functional.pad(tile, (0, pad_w, 0, pad_h), mode='reflect') if pad_h > 0 or pad_w > 0 else tile

                    with torch.no_grad():
                        if self.device.type == 'cuda':
                            with torch.amp.autocast('cuda'):
                                out_padded = self.model(tile_padded)
                        else:
                            out_padded = self.model(tile_padded)
                        out_padded = torch.clamp(out_padded, 0.0, 1.0)

                    out_tile = out_padded[:, :, :th, :tw].cpu()

                    output[:, :, y1:y2, x1:x2] += out_tile
                    count[:, :, y1:y2, x1:x2] += 1.0

            denoised_final = output / count

        denoised_pil = transforms.ToPILImage()(denoised_final.squeeze(0))
        return img_pil, denoised_pil

    def save_result(self, image_path: str, denoised_pil: Image.Image) -> str:
        """
        Сохраняет обработанное изображение в подпапку 'results' (создается автоматически
        рядом со скриптом, из которого вызван класс).

        :param image_path: Путь к оригинальному файлу (используется для формирования имени нового файла).
        :param denoised_pil: Объект очищенного изображения для сохранения.

        :return: str: Абсолютный путь к сохраненному файлу.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        base_name = os.path.basename(image_path)
        file_name, ext = os.path.splitext(base_name)
        save_path = os.path.join(results_dir, f"{file_name}_denoised.png")

        denoised_pil.save(save_path)
        print(f"Результат сохранен в: {save_path}")
        return save_path

    def process_and_save(self, input_path: str, output_path: str) -> None:
        """
        Берет изображение по input_path, прогоняет через нейросеть
        и сохраняет очищенный результат по строго заданному output_path.

        :param input_path: Путь к исходному (зашумленному) файлу.
        :param output_path: Полный путь, куда нужно сохранить результат (включая имя файла и расширение).
        """
        if not os.path.exists(input_path):
            print(f"Исходный файл не найден: {input_path}")
            return

        try:
            _, clean_pil = self.process_image(input_path)

            output_dir = os.path.dirname(os.path.abspath(output_path))
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            clean_pil.save(output_path)
            print(f"Файл сохранен: {output_path}")

        except Exception as e:
            print(f"Сбой при обработке {input_path}: {e}")

    @staticmethod
    def show_interactive_slider(img_noisy_pil: Image.Image, img_clean_pil: Image.Image):
        """
        Утилита для визуального тестирования. Открывает графическое окно Matplotlib
        с интерактивным ползунком "до/после" для сравнения картинок.

        ВНИМАНИЕ: Блокирует выполнение программы до закрытия окна.

        :param img_noisy_pil: Оригинальное (зашумленное) изображение.
        :param img_clean_pil: Очищенное изображение.
        """
        img_noisy_np = np.array(img_noisy_pil)
        img_clean_np = np.array(img_clean_pil)

        fig, ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.15)

        initial_split = 0.5
        width = img_noisy_np.shape[1]
        split_idx = int(initial_split * width)

        combined_img = np.copy(img_clean_np)
        combined_img[:, :split_idx] = img_noisy_np[:, :split_idx]

        im_display = ax.imshow(combined_img)
        v_line = ax.axvline(x=split_idx, color='red', linestyle='--', linewidth=2)

        ax.axis('off')
        ax.set_title("Сдвигайте ползунок: <-- ШУМНОЕ | ОЧИЩЕННОЕ -->\n(Закройте окно для продолжения)",
                     fontsize=12, fontweight='bold')

        ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
        slider = Slider(ax_slider, 'Разделитель', 0.0, 1.0, valinit=initial_split, color='cyan')

        def update(val):
            split_idx = int(val * width)
            new_combined = np.copy(img_clean_np)
            new_combined[:, :split_idx] = img_noisy_np[:, :split_idx]
            im_display.set_data(new_combined)
            v_line.set_xdata([split_idx, split_idx])
            fig.canvas.draw_idle()

        slider.on_changed(update)
        fig.slider_ref = slider

        print("Окно с результатом открыто.")
        plt.show()


if __name__ == '__main__':
    print(" ИНТЕРАКТИВНОЕ ТЕСТИРОВАНИЕ: UNET DENOISER")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, "models", "best_weight.pth")

    try:
        denoiser = ImageDenoiser(weights_path)
    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА {e}")
        input("Нажмите Enter для выхода...")
        exit()

    while True:
        image_path = input("Введите путь к картинке (или 'q' для выхода): ").strip().strip('"').strip("'")

        if image_path.lower() == 'q':
            print("Завершение работы. Удачи!")
            break

        if not os.path.exists(image_path):
            print(f"Изображение '{image_path}' не найдено.")
            continue

        try:
            original_img, clean_img = denoiser.process_image(image_path)
            denoiser.save_result(image_path, clean_img)
            denoiser.show_interactive_slider(original_img, clean_img)

        except Exception as e:
            print(f"Произошла ошибка при обработке: {e}")
