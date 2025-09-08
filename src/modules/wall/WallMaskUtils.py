import cv2
import numpy as np

# ------------------------
# Морфология
# ------------------------

def apply_morphology(mask, kernel_size=5):
    """
    Применяет морфологические операции для утолщения и очистки стен:
    - дилатация → эрозия → дилатация.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def connect_close_walls(mask, kernel_size=20):
    """
    Соединяет близко расположенные сегменты стен.
    Использует морфологическую операцию closing.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


# ------------------------
# Контуры
# ------------------------

def filter_contours(mask, min_area=750, min_perimeter=500, min_aspect_ratio=1.5, max_aspect_ratio=20):
    """
    Фильтрует найденные контуры:
    - исключает мелкие по площади;
    - исключает по периметру;
    - исключает слишком вытянутые или слишком компактные фигуры.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned_mask = np.zeros_like(mask)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / min(w, h)
            perimeter = cv2.arcLength(contour, True)

            if min_aspect_ratio < aspect_ratio < max_aspect_ratio or perimeter > min_perimeter:
                cv2.drawContours(cleaned_mask, [contour], -1, 255, thickness=cv2.FILLED)

    return cleaned_mask


def close_contour(contour):
    """
    Замыкает контур, если первая и последняя точки не совпадают.
    """
    if np.linalg.norm(contour[0] - contour[-1]) > 1:
        contour = np.append(contour, [contour[0]], axis=0)
    return contour


def get_mask_contours(mask):
    """
    Находит и возвращает список замкнутых контуров из маски.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    closed_contours = [close_contour(c) for c in contours]
    closed_contours = map(np.squeeze, closed_contours)
    return list(closed_contours)


# ------------------------
# Упрощение контуров
# ------------------------

def distance(p1, p2):
    """Евклидово расстояние между двумя точками."""
    return np.linalg.norm(np.array(p1) - np.array(p2))


def is_collinear(p1, p2, p3, collinearity_threshold=15):
    """
    Проверяет, лежат ли три точки почти на одной линии.
    Используется площадь треугольника как мера коллинеарности.
    """
    area = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
    return area < collinearity_threshold


def simplify_contour(contour, distance_threshold=25, collinearity_threshold=15):
    """
    Упрощает один контур:
    - убирает слишком близкие точки;
    - убирает коллинеарные точки;
    - сохраняет форму стены.
    """
    if len(contour) < 3:
        return None

    simplified = [contour[0]]
    for i in range(1, len(contour) - 1):
        prev_point = simplified[-1]
        curr_point = contour[i]
        next_point = contour[i + 1]

        if (
            distance(prev_point, curr_point) > distance_threshold
            or not is_collinear(prev_point, curr_point, next_point, collinearity_threshold)
        ):
            simplified.append(curr_point)

    simplified.append(contour[-1])

    if len(simplified) > 2:
        return np.array(simplified, dtype=int)
    return None


def simplify_contours(contours, distance_threshold=25, collinearity_threshold=15):
    """
    Упрощает список контуров, вызывая simplify_contour для каждого.
    """
    simplified = []
    for contour in contours:
        result = simplify_contour(contour, distance_threshold, collinearity_threshold)
        if result is not None:
            simplified.append(result)
    return simplified


# ------------------------
# Комплексные пайплайны
# ------------------------

def clean_mask(mask, min_area=750, min_perimeter=500, min_aspect_ratio=1.5, max_aspect_ratio=20):
    """
    Полная очистка маски:
    1. Морфология.
    2. Фильтрация контуров.
    3. Соединение близких стен.
    """
    mask = apply_morphology(mask)
    mask = filter_contours(mask, min_area, min_perimeter, min_aspect_ratio, max_aspect_ratio)
    mask = connect_close_walls(mask)
    return mask
