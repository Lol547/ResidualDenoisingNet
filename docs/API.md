
# API Reference — ImageDenoiser

Класс `ImageDenoiser` предоставляет простой интерфейс для загрузки модели и обработки изображений. Вся сложность (CUDA/CPU, паддинг, смешанная точность, тайлинг) скрыта внутри методов.

---

## Конструктор

### `__init__(weights_path: str)`

Инициализирует нейросеть. Автоматически определяет доступность видеокарты (CUDA/CPU), переносит модель на нужное устройство и переводит её в режим `eval()`.

**Параметры:**
| Имя | Тип | Описание |
|-----|-----|----------|
| `weights_path` | `str` | Путь к файлу весов модели (`.pth`) |

**Исключения:**
- `FileNotFoundError` - если файл весов не найден

**Пример:**
```python
from inference import ImageDenoiser

denoiser = ImageDenoiser("checkpoints/best_weight.pth")
```

---

## Основные методы

### `process_image(image_path: str, tile_size: int = 800, tile_overlap: int = 64) -> tuple[Image.Image, Image.Image]`

Главный метод инференса. Берёт на себя всю работу с тензорами: автоматически добавляет временный паддинг до кратности 32 (ограничение архитектуры UNet), прогоняет через сеть (с `amp.autocast` для ускорения на GPU) и обрезает обратно до исходного размера.

**Параметры:**
| Имя | Тип | Описание |
|-----|-----|----------|
| `image_path` | `str` | Путь к исходному изображению на диске |
| `tile_size` | `int` | Максимальный размер тайла для обработки по частям (по умолчанию 800) |
| `tile_overlap` | `int` | Размер нахлёста между тайлами (скрывает швы при склейке, по умолчанию 64) |

**Возвращает:**
- Кортеж из двух объектов `PIL.Image`: `(оригинал, очищенное_изображение)`

**Пример:**
```python
original, denoised = denoiser.process_image("test_images/noisy_photo.jpg")
denoised.save("output/clean_photo.png")
```

---

### `save_result(image_path: str, denoised_pil: Image.Image) -> str`

Вспомогательный метод для сохранения. Автоматически создаёт папку `results` (в текущей директории) и сохраняет изображение, добавляя к оригинальному имени файла суффикс `_denoised`.

**Параметры:**
| Имя | Тип | Описание |
|-----|-----|----------|
| `image_path` | `str` | Путь к исходному файлу (используется для парсинга имени) |
| `denoised_pil` | `Image.Image` | Объект очищенного изображения (из `process_image`) |

**Возвращает:**
- `str` - абсолютный путь к сохранённому файлу

**Пример:**
```python
saved_path = denoiser.save_result("test_images/noisy_photo.jpg", denoised)
print(f"Результат сохранён: {saved_path}")
# Результат: results/noisy_photo_denoised.png
```

---

### `process_and_save(input_path: str, output_path: str) -> None`

Метод "всё в одном": загружает изображение, прогоняет через нейросеть и сразу сохраняет результат по указанному пути. Автоматически создаёт папки для выходного файла, если их нет.

**Параметры:**
| Имя | Тип | Описание |
|-----|-----|----------|
| `input_path` | `str` | Путь к исходному (зашумлённому) файлу |
| `output_path` | `str` | Полный путь для сохранения результата (включая имя файла и расширение) |

**Возвращает:**
- `None` - результат сохраняется на диск

**Пример:**
```python
denoiser.process_and_save(
    input_path="photos/raw_image.jpg",
    output_path="photos/processed/clean_image.png"
)
```

---

## Визуализация

### `show_interactive_slider(img_noisy_pil: Image.Image, img_clean_pil: Image.Image)`

Утилита для визуального сравнения. Открывает окно `matplotlib` с интерактивным ползунком "до/после".

**Важно:** Блокирует выполнение программы до закрытия окна.

**Параметры:**
| Имя | Тип | Описание |
|-----|-----|----------|
| `img_noisy_pil` | `Image.Image` | Оригинальное (зашумлённое) изображение |
| `img_clean_pil` | `Image.Image` | Очищенное изображение |

**Пример:**
```python
original, denoised = denoiser.process_image("test_images/noisy_photo.jpg")
denoiser.show_interactive_slider(original, denoised)
```

---

## Структура выходных файлов

| Метод | Папка сохранения | Имя файла |
|-------|------------------|-----------|
| `save_result()` | `results/` (создаётся автоматически) | `{original_name}_denoised.png` |
| `process_and_save()` | Указывается пользователем | Указывается пользователем |

---

## Полный пример работы

```python
from inference import ImageDenoiser

# 1. Инициализация
denoiser = ImageDenoiser("checkpoints/best_weight.pth")

# 2. Обработка
original, clean = denoiser.process_image("my_photo.jpg")

# 3. Сохранение с суффиксом _denoised
denoiser.save_result("my_photo.jpg", clean)

# 4. Визуальное сравнение
denoiser.show_interactive_slider(original, clean)
```

---

## Примечания

- **Поддерживаемые форматы**: `.jpg`, `.png`, `.bmp`, `.tiff`
- **Формат результата**: сохраняется в `.png` без потерь (для максимального качества)
- **GPU**: автоматически используется CUDA, если доступен
- **Тайлинг**: автоматически включается для изображений > 800x800

---

## Требования

- Python 3.8+
- PyTorch
- См. полный список в `requirements.txt`
