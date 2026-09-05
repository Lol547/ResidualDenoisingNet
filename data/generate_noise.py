"""
generate_noise.py - генерация смешанного шума для чистых изображений.

Применяет случайную комбинацию гауссовского шума, пуассоновского шума и JPEG-артефактов.
Используется для создания зашумлённых версий изображений DIV2K.

Использование:
    python generate_noise.py --input_dir /path/to/DIV2K_train_HR --output_dir /path/to/DIV2K_train_HR_noisy
"""

import os
import cv2
import numpy as np
import random
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime


def setup_logging(output_folder):
    """Настраивает логирование."""
    log_folder = os.path.join(output_folder, "logs")
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_folder, f"noise_generation_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_file


def apply_gaussian_noise(image, sigma):
    """Добавляет гауссовский шум с заданным стандартным отклонением."""
    noise = np.random.normal(0, sigma, image.shape).astype(np.uint8)
    return cv2.add(image, noise)


def apply_poisson_noise(image):
    """Добавляет пуассоновский шум (сигнал-зависимый)."""
    image_float = image.astype(np.float64) / 255.0
    noisy = np.random.poisson(image_float * 255) / 255.0
    return (noisy * 255).astype(np.uint8)


def apply_jpeg_compression(image, quality):
    """Добавляет артефакты JPEG-сжатия."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(encimg, cv2.IMREAD_COLOR)


def add_random_noise(image):
    """
    Случайно выбирает типы шумов и применяет их к изображению.

    Возможные шумы: гауссовский (σ от 2 до 30), пуассоновский, JPEG (quality 10-40).
    Каждое изображение получает 1-3 типа шума.
    """
    noisy = image.copy()
    applied_noises = []
    params = {}

    if random.random() > 0.3:
        sigma = random.randint(2, 30)
        noisy = apply_gaussian_noise(noisy, sigma)
        applied_noises.append(f"Gaussian(sigma={sigma})")
        params['gaussian_sigma'] = sigma

    if random.random() > 0.5:
        noisy = apply_poisson_noise(noisy)
        applied_noises.append("Poisson")
        params['poisson'] = True

    if random.random() > 0.4:
        quality = random.randint(10, 40)
        noisy = apply_jpeg_compression(noisy, quality)
        applied_noises.append(f"JPEG(quality={quality})")
        params['jpeg_quality'] = quality

    if not applied_noises:
        sigma = random.randint(2, 30)
        noisy = apply_gaussian_noise(noisy, sigma)
        applied_noises.append(f"Gaussian(sigma={sigma})")
        params['gaussian_sigma'] = sigma

    return noisy, applied_noises, params


def main():
    parser = argparse.ArgumentParser(description="Генерация смешанного шума для чистых изображений.")
    parser.add_argument('--input_dir', type=str, required=True, help='Папка с чистыми изображениями')
    parser.add_argument('--output_dir', type=str, required=True, help='Папка для сохранения зашумлённых изображений')
    parser.add_argument('--extension', type=str, default='.png', help='Расширение выходных файлов (по умолчанию .png)')
    args = parser.parse_args()

    log_file = setup_logging(args.output_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logging.info(f"Исходная папка: {args.input_dir}")
    logging.info(f"Выходная папка: {args.output_dir}")

    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    stats = {
        'total_processed': 0,
        'successful': 0,
        'errors': 0,
        'noise_stats': {},
        'processed_files': []
    }

    for filename in os.listdir(args.input_dir):
        if not filename.lower().endswith(image_extensions):
            continue

        file_path = os.path.join(args.input_dir, filename)
        stats['total_processed'] += 1

        try:
            image = cv2.imread(file_path)
            if image is None:
                raise Exception("Не удалось прочитать изображение")

            noisy_image, applied_noises, params = add_random_noise(image)

            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(args.output_dir, f"{base_name}{args.extension}")
            cv2.imwrite(output_path, noisy_image)

            noise_str = " + ".join(applied_noises)
            logging.info(f"Обработано: {filename} -> {noise_str}")

            stats['successful'] += 1
            stats['processed_files'].append({
                'original': filename,
                'output': os.path.basename(output_path),
                'noises': applied_noises,
                'params': params
            })

            for noise in applied_noises:
                base_noise = noise.split('(')[0]
                stats['noise_stats'][base_noise] = stats['noise_stats'].get(base_noise, 0) + 1

        except Exception as e:
            logging.error(f"Ошибка при обработке {filename}: {e}")
            stats['errors'] += 1

    report_path = os.path.join(args.output_dir, "noise_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logging.info(f"Отчёт сохранён: {report_path}")
    print(f"\nОбработано изображений: {stats['successful']} / {stats['total_processed']}")
    print(f"Ошибок: {stats['errors']}")
    print(f"Отчёт: {report_path}")


if __name__ == '__main__':
    main()
