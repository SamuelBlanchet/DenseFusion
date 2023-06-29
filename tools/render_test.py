import os
# switch to "osmesa" or "egl" before loading pyrender
#os.environ["PYOPENGL_PLATFORM"] = "osmesa"
#os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import pyrender
import trimesh
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Uncolored mesh
#part_trimesh = trimesh.load("../datasets/exo/Exo_preprocessed/objects/259.stl")
part_trimesh = trimesh.load("../datasets/exo/Exo_preprocessed/models/obj_05.ply")       # [-0.8830567, -0.5400259, -0.7] pour environ 0
part_trimesh.visual.vertex_colors = np.full(part_trimesh.vertices.shape, [255, 0, 0])
part_2_trimesh = trimesh.load("../datasets/exo/Exo_preprocessed/models/obj_06.ply")       # [-0.8830567, -0.5400259, -0.7] pour environ 0
part_2_trimesh.visual.vertex_colors = np.full(part_trimesh.vertices.shape, [0, 0, 255])

# Colored mesh
#part_trimesh = trimesh.load("../datasets/exo/Exo_preprocessed/objects/reservoir_transparent_A_colored.ply")

# Shpere from example
sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.8)
sphere.vertices+=1e-2*np.random.randn(*sphere.vertices.shape)


#mesh = pyrender.Mesh.from_trimesh(sphere, smooth=False)
mesh = pyrender.Mesh.from_trimesh(part_trimesh, smooth=False)
mesh_2 = pyrender.Mesh.from_trimesh(part_2_trimesh, smooth=False)


# compose scene
scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[100, 100, 100])
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
light = pyrender.DirectionalLight(color=[1,1,1], intensity=2e3)

scene.add(mesh, pose=np.eye(4))
scene.add(mesh_2, pose=np.eye(4))
scene.add(light, pose=np.eye(4))

#c = 2**-0.5
#scene.add(camera, pose=[[ 1,  0,  0,  0],
#                        [ 0,  c, -c, -2],
#                        [ 0,  c,  c,  2],
#                        [ 0,  0,  0,  1]])
matrix=np.array([np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)])
matrix[:,3]=np.array([88.30567, 54.00259, 200, 1])     # Positions x, y, z  ,         [-0.8830567, -0.5400259, -0.7]
rotation = R.from_euler('xyz', [0, 0, 0], degrees=True)
matrix[0:3,0:3]=rotation.as_matrix()
scene.add(camera, pose=matrix)

# render scene
pyrender.Viewer(scene)
r = pyrender.OffscreenRenderer(1024, 1024)
color, _ = r.render(scene)
plt.imsave("test.png", color)