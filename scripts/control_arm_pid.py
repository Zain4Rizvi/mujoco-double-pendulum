import mujoco
import mujoco.viewer
import math

model = mujoco.MjModel.from_xml_path("../models/arm.xml")
data = mujoco.MjData(model)

integral_1 = 0
integral_2 = 0.0
integral_3 = 0.0
integral_limit = 20.0  # anti-windup clamp, tune this

prev_error2 = 0.0
prev_error3 = 0.0
curr_time = 0


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
        # kp1 = -30
        # ki1 = -5
        # kd1 = -10

        # kp2 = -15
        # ki2 = -2
        # kd2 = -10

        kp1 = 10
        ki1 = 0
        kd1 = 1000
        speed = 0.5

        kp2 = -20
        ki2 = 0
        kd2 = -1

        kp3 = -5
        ki3 = -0.1
        kd3 = -1


        # Desired Values
        error1 = curr_time * speed - p1
        error2 = (p2 + p1)   % (2 * math.pi) - math.pi
        error3 = p3

        #Compute Values
        integral_1 += dt * error1
        integral_2 = max(-integral_limit, min(integral_limit, integral_1))
        integral_2 += dt * error2
        integral_2 = max(-integral_limit, min(integral_limit, integral_2))
        integral_3 += dt * error3
        integral_3 = max(-integral_limit, min(integral_limit, integral_3))
        print(integral_3)

        derivative_1 = speed - v1
        derivative_2 = v2
        derivative_3 = v3

        # Modifying Joint 2
        update_1 = kp1 * error1 + ki1 * integral_1 + kd1 * derivative_1
        update_2 = kp2 * error2 + ki2 * integral_2 + kd2 * derivative_2
        update_3 = kp3 * error3 + ki3 * integral_3 + kd3 * derivative_3
        #print(error1,update[0], update_2[0])
        # Update Torque for Arm 1 and Arm 2
  
        data.ctrl[0] = update_1[0]
        data.ctrl[1] = update_2[0]
        data.ctrl[2] = update_3[0]

        #mujoco.mj_forward(model, data)   # NOT mj_step — updates kinematics without integrating

        mujoco.mj_step(model, data)



        # Update "previous" state for next iteration's derivative
        curr_time += dt

        viewer.sync()