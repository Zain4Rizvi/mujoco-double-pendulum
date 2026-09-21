import mujoco
import mujoco.viewer
import math

model = mujoco.MjModel.from_xml_path("../models/arm.xml")
data = mujoco.MjData(model)


with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():
        p1 = data.joint('joint1').qpos
        v1 = data.joint('joint1').qvel
        a1 = data.joint('joint1').qacc

        p2 = data.joint('joint2').qpos
        v2 = data.joint('joint2').qvel
        a2 = data.joint('joint2').qacc


        # # Apply torque to joint 1
        # data.ctrl[0] = 100

        # # Don't apply torque to joint 2
        # data.ctrl[1] = -3
        data.joint('joint1').qpos = math.pi
        mujoco.mj_forward(model, data)   # NOT mj_step — updates kinematics without integrating

        #mujoco.mj_step(model, data)

        print("qpos:", data.qpos)

        viewer.sync()