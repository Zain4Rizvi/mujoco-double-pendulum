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

        p3 = data.joint('joint3').qpos
        v3 = data.joint('joint3').qvel
        a3 = data.joint('joint3').qacc
        

        print(p1,v1,a1)


        # # Apply torque to joint 1

        time = model.opt.timestep
        k = 0.0004
        desired_angle = math.pi - p1
        update = k * (desired_angle - p2 - v2 * time) * (2/(time**2))


        k2 = 0.00002
        desired_angle_2 = 0
        update_2 = k2 * (desired_angle_2 - p3 - v3 * time) * (2/(time**2))

        # Update Torque for Arm 1 and Arm 2
        data.ctrl[0] = 100
        data.ctrl[1] = update
        data.ctrl[2] = update_2

        #mujoco.mj_forward(model, data)   # NOT mj_step — updates kinematics without integrating

        mujoco.mj_step(model, data)


        viewer.sync()