from sklearn.cluster import KMeans
from collections import Counter

def get_dominant_color(image):
    """
    Находит доминантный цвет изображения (основной цвет стен) через KMeans.
    """
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(image)

    colors = kmeans.cluster_centers_
    counts = Counter(kmeans.labels_)
    dominant_color = colors[counts.most_common(1)[0][0]]
    return dominant_color


def filter_non_white_pixels(image_rgb):
    """
    Убирает белые пиксели (фон) из изображения.
    """
    mask_non_white = (image_rgb[:, :, 0] < 220) | (image_rgb[:, :, 1] < 220) | (image_rgb[:, :, 2] < 220)
    return image_rgb[mask_non_white]
