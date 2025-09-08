import os
import numpy as np
from PIL import Image
from src.modules.wall.WallMaskExtractor import WallExtractor, get_mask_contours
from src.modules.scene.SceneGenerator import SceneGenerator, screenshot_scene, export_scene

os.environ['LOKY_MAX_CPU_COUNT'] = '4'

wall_extractor = WallExtractor()
scene_generator = SceneGenerator()

# Путь к папке с изображениями и папке для сохранения результатов
input_folder = '../datasets/test_images'
output_mask_folder = '../output/wall_mask'
output_scene_folder = '../output/scene'

if not os.path.exists(output_mask_folder):
    os.makedirs(output_mask_folder)

if not os.path.exists(output_scene_folder):
    os.makedirs(output_scene_folder)

# Перебор всех изображений в папке
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        input_image_path = os.path.join(input_folder, filename)

        wall_mask = wall_extractor.extract(input_image_path)
        # Изображение маски
        output_image = Image.fromarray(wall_mask.astype(np.uint8))

        output_image_path = os.path.join(output_mask_folder, filename)
        output_image.save(output_image_path)

        # 3D сцена
        wall_contours = get_mask_contours(wall_mask)
        scene = scene_generator.generate(wall_contours)

        screenshot_scene(scene, output_scene_folder, filename)
        export_scene(scene, output_scene_folder, filename)

        print(f'Обработано и сохранено: {filename}')
