import mujoco
import mujoco.viewer
import math
import numpy as np
from numpy.linalg import matrix_rank
from scipy.linalg import solve_continuous_are

model = mujoco.MjModel.from_xml_path("../models/arm.xml")
data = mujoco.MjData(model)

def wrap_to_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

nx = 4   # state size: [e2, e3, v2, v3]
nu = 2   # input size: [tau2, tau3]  (joint1 is driven separately, not by LQR)

# Equilibrium in ERROR coordinates — always zero, regardless of p1's motion
x_star = np.zeros(nx)
u_star = np.zeros(nu)

# A, B computed numerically at the equilibrium (fill in via mjd_transitionFD,
# then extract just the joint2/joint3 rows+cols — see note below)
A_full = np.zeros((2 * model.nv, 2 * model.nv))
B_full = np.zeros((2 * model.nv, model.nu))
epsilon = 1e-6
mujoco.mjd_transitionFD(model, data, epsilon, True, A_full, B_full, None, None)

# Cost weights — YOUR choice, tune by feel
Q = np.diag([10.0, 10.0, 1.0, 1.0])   # penalize e2, e3 angle error most
R = np.diag([1.0, 1.0])               # penalize torque effort

#P = solve_continuous_are(A, B, Q, R)
#K = np.linalg.inv(R) @ B.T @ P         # shape (nu, nx) = (2, 4)

data.qpos[0] = math.pi
data.qpos[1] = 0
data.qpos[2] = 0
data.qvel[0] = 0
data.qvel[1] = 0
data.qvel[2] = 0

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():
        p1 = data.joint('joint1').qpos[0]
        v1 = data.joint('joint1').qvel[0]
        p2 = data.joint('joint2').qpos[0]
        v2 = data.joint('joint2').qvel[0]
        p3 = data.joint('joint3').qpos[0]
        v3 = data.joint('joint3').qvel[0]

        def wrap_to_pi(a):
            return (a + math.pi) % (2 * math.pi) - math.pi

        # e2 = wrap_to_pi(p2 - (math.pi - p1))   # error state, wrapped
        # e3 = wrap_to_pi(p3)
        # de2 = v2 + v1                           # d/dt of e2 (chain rule, matches earlier fix)
        # de3 = v3

        # x = np.array([e2, e3, de2, de3])
        # delta_x = x - x_star                    # = x, since x_star is zero, but keep the form

        # u = u_star - K @ delta_x                # shape (2,) -> [tau2, tau3]

        # data.ctrl[1] = u[0]
        # data.ctrl[2] = u[1]
        # data.ctrl[0] still driven by your open-loop oscillator, unchanged
        




        # Update Torque for Arm 1 and Arm 2
        # if curr_time % 50 <= 25:
        #     data.ctrl[0] = 100
        # else:
        #     data.ctrl[0] = -100
        # data.ctrl[1] = 0
        # data.ctrl[2] = 0

        #mujoco.mj_forward(model, data)   # NOT mj_step — updates kinematics without integrating

        mujoco.mj_step(model, data)



        # Update "previous" state for next iteration's derivative

        viewer.sync()