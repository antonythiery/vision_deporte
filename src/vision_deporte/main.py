import numpy as np
import os
import time
import mujoco
import mujoco.viewer
import xml.etree.ElementTree as ET

model = mujoco.MjModel.from_xml_path("src/vision_deporte/models/robot.xml")
data = mujoco.MjData(model)

mujoco.mj_kinematics(model, data)

viewer = mujoco.viewer.launch(
    model = model,
    data = data,
    show_left_ui=False,
    show_right_ui=False,
)

try :
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
except KeyboardInterrupt: 
    pass
finally : 
    viewer.close()