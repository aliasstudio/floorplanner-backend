import logging
import os
import numpy as np
import pyvista as pv
import triangle as tr
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

logging.basicConfig(level=logging.INFO)

def export_scene(scene, path, filename = 'scene'):
    filename = os.path.splitext(filename)[0]

    os.makedirs(path, exist_ok=True)
    scene.save(os.path.join(path, filename + '.stl'))
    scene.save(os.path.join(path, filename + '.obj'))

def screenshot_scene(scene, path, filename = 'scene'):
    plotter = pv.Plotter(off_screen=True)

    plotter.add_mesh(scene, color="grey", show_edges=True)
    plotter.view_isometric()

    filename = os.path.splitext(filename)[0]

    os.makedirs(path, exist_ok=True)
    plotter.screenshot(os.path.join(path, filename + '.png'))
    plotter.close()

def preprocess_contour(contour):
    polygon = Polygon(contour)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    polygon = orient(polygon, sign=1.0)
    return np.array(polygon.exterior.coords)

def remove_duplicate_points(contour):
    _, unique_indices = np.unique(np.round(contour, decimals=5), axis=0, return_index=True)
    return contour[np.sort(unique_indices)]

class SceneGenerator:
    def __init__(self, wall_height=2.7 * 50):
        self.wall_height = wall_height

    def generate(self, wall_mask):
        scene = pv.PolyData()

        for contour in wall_mask:
            try:
                # Предварительная обработка
                contour = preprocess_contour(contour)
                contour = remove_duplicate_points(contour)

                if len(contour) < 3:
                    logging.warning(f"Контур слишком мал для триангуляции: {contour}")
                    continue

                # Создаем сегменты
                num_points = len(contour)
                segments = np.array([[i, (i + 1) % num_points] for i in range(num_points)])

                input_data = {'vertices': contour, 'segments': segments}
                tri = tr.triangulate(input_data, 'p')

                if 'triangles' not in tri or tri['triangles'] is None:
                    logging.warning(f"Не удалось выполнить триангуляцию для контура: {contour}")
                    continue

                tri_points = np.array(tri['vertices'])
                tri_faces = np.array(tri['triangles'])

                tri_points_3d = np.hstack((tri_points, np.zeros((tri_points.shape[0], 1))))

                faces = []
                for triangle in tri_faces:
                    faces.append(3)
                    faces.extend(triangle)

                faces = np.array(faces)
                wall = pv.PolyData(tri_points_3d, faces)
                wall = wall.extrude((0, 0, self.wall_height), capping=True)

                scene += wall
            except Exception as e:
                logging.error(f"Ошибка при обработке контура: {contour}. Exception: {e}")
                continue

        scene = scene.clean()
        scene = scene.compute_normals()

        # scene = scene.decimate_pro(reduction=0.5, preserve_topology=True)
        # scene = scene.decimate(target_reduction=0.5)

        return scene
