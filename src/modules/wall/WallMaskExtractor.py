import cv2
import numpy as np

from WallMaskUtils import (
    clean_mask,
    get_mask_contours,
    simplify_contours,
)
from ..utils.ImageUtils import filter_non_white_pixels, get_dominant_color


class WallExtractor:
    def __init__(self, min_area=750, min_perimeter=500, min_aspect_ratio=1.5, max_aspect_ratio=20):
        self.min_area = min_area
        self.min_perimeter = min_perimeter
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

    def extract(self, image_path):
        """
        Выполняет пошаговый пайплайн:
        1. Загружает изображение.
        2. Убирает белый фон.
        3. Определяет доминантный цвет (стены).
        4. Строит бинарную маску стен.
        5. Очищает маску (морфология → фильтрация контуров → объединение).
        6. Извлекает и упрощает контуры.

        :param image_path: путь к изображению
        :return: список контуров
        """
        # 1. Читаем изображение
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Убираем белый фон
        filtered_image = filter_non_white_pixels(image_rgb)

        # 3. Находим основной цвет стен
        dominant_color = get_dominant_color(filtered_image)

        # 4. Создаем маску для пикселей, близких к доминантному цвету
        lower_bound = np.clip(dominant_color - 10, 0, 255)
        upper_bound = np.clip(dominant_color + 25, 0, 255)
        wall_mask = cv2.inRange(image_rgb, lower_bound, upper_bound)

        # 5. Чистим маску стен
        mask = clean_mask(
            wall_mask,
            min_area=self.min_area,
            min_perimeter=self.min_perimeter,
            min_aspect_ratio=self.min_aspect_ratio,
            max_aspect_ratio=self.max_aspect_ratio,
        )

        # 6. Контуры
        contours = get_mask_contours(mask)
        contours = simplify_contours(contours)

        return contours
