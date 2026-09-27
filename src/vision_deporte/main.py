import numpy as np
import os
import mujoco
import mediapy as media

from src.vision_deporte.detection import detectArucos

image_dir = "datas/calibration_images"

images = [
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.endswith(".png") or f.endswith(".jpg")
]

cameraParams = np.load("src/vision_deporte/cameraParams.npz")
detectorBasedeporte = detectArucos(images[1], cameraParams)

# print(detectorBasedeporte.detect())

model = mujoco.MjModel.from_xml_path("src/vision_deporte/models/robot.xml")
data = mujoco.MjData(model)

mujoco.mj_kinematics(model, data)

with mujoco.Renderer(model) as renderer:
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
  
    media.show_image(renderer.render())