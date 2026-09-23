import mujoco
import mujoco.viewer
import math

model = mujoco.MjModel.from_xml_path("../models/arm.xml")
data = mujoco.MjData(model)

integral_1 = 0.0
integral_2 = 0.0
integral_limit = 10.0  # anti-windup clamp, tune this

prev_error1 = 0.0
prev_error2 = 0.0


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
        

        # Define Constants
        dt = model.opt.timestep
        kp1 = -30
        ki1 = -2
        kd1 = -10

        kp2 = -15
        ki2 = -2
        kd2 = -10

        # Desired Values
        error1 = (p2 + p1)   % (2 * math.pi) - math.pi
        error2 = p3

        #Compute Values
        integral_1 += dt * error1
        integral_1 = max(-integral_limit, min(integral_limit, integral_1))
        integral_2 += dt * error2
        integral_2 = max(-integral_limit, min(integral_limit, integral_2))


        derivative_1 = v2
        derivative_2 = (error2 - prev_error2) / dt

        # Modifying Joint 2

        update = kp1 * error1 + ki1 * integral_1 + kd1 * derivative_1

        # Modifying Joint 3
        
        update_2 = kp2 * error2 + ki2 * integral_2 + kd2 * derivative_2
        print(error1,update[0], update_2[0])
        # Update Torque for Arm 1 and Arm 2
        data.ctrl[0] = 100
        data.ctrl[1] = update[0]
        data.ctrl[2] = update_2[0]

        #mujoco.mj_forward(model, data)   # NOT mj_step — updates kinematics without integrating

        mujoco.mj_step(model, data)



        # Update "previous" state for next iteration's derivative
        prev_error1 = error1
        prev_error2 = error2

        viewer.sync()