import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
import time

XML = """
<mujoco model="neuro_grasp_arm">
    <compiler angle="radian" coordinate="local"/>
    <option gravity="0 0 -9.81" timestep="0.01"/>

    <asset>
        <texture type="2d" name="grid" builtin="checker" rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4" width="512" height="512"/>
        <material name="mat_floor" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0.2"/>
        <material name="mat_robot" rgba="0.9 0.9 0.9 1"/>
        <material name="mat_joint" rgba="0.2 0.2 0.2 1"/>
        <material name="mat_target" rgba="0.9 0.1 0.1 1"/>
    </asset>

    <worldbody>
        <light pos="0 0 3" dir="0 0 -1" directional="true" castshadow="true"/>
        <geom name="floor" type="plane" size="2 2 0.1" material="mat_floor"/>
        <geom name="table" type="box" size="0.4 0.4 0.2" pos="0 0 0.2" rgba="0.3 0.3 0.3 1"/>

        <body name="target_block" pos="0.25 0 0.43">
            <freejoint/>
            <geom type="box" size="0.03 0.03 0.03" material="mat_target" mass="0.1" friction="1 1 1"/>
        </body>

        <body name="base" pos="0 0 0.4">
            <geom type="cylinder" size="0.06 0.02" material="mat_joint"/>
            <body name="shoulder" pos="0 0 0.02">
                <joint name="j1_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                <geom type="capsule" size="0.035" fromto="0 0 0 0 0 0.25" material="mat_robot"/>
                <body name="elbow" pos="0 0 0.25">
                    <joint name="j2_lift" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
                    <geom type="cylinder" size="0.04 0.035" axisangle="1 0 0 1.57" material="mat_robot"/>
                    <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.25" material="mat_robot"/>
                    <body name="wrist" pos="0 0 0.25">
                        <joint name="j3_flex" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
                        <geom type="cylinder" size="0.035 0.03" axisangle="1 0 0 1.57" material="mat_robot"/>
                        <geom type="capsule" size="0.025" fromto="0 0 0 0 0 0.15" material="mat_robot"/>
                        <body name="ee" pos="0 0 0.15">
                            <geom type="sphere" size="0.03" material="mat_joint"/>
                        </body>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>

    <actuator>
        <position name="motor1" joint="j1_pan" kp="20" ctrllimited="true" ctrlrange="-3.14 3.14"/>
        <position name="motor2" joint="j2_lift" kp="20" ctrllimited="true" ctrlrange="-3.14 3.14"/>
        <position name="motor3" joint="j3_flex" kp="20" ctrllimited="true" ctrlrange="-3.14 3.14"/>
    </actuator>
</mujoco>
"""


class BehaviorCloningPolicy(nn.Module):
    def __init__(self, input_dim=6, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def main():
    print("🧠 Loading Trained PyTorch Behavior Cloning Policy...")
    policy = BehaviorCloningPolicy()
    policy.load_state_dict(torch.load("models/bc_policy.pth"))
    policy.eval()

    print("🤖 Initializing MuJoCo Simulation...")
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for episode in range(5):
            print(f"\n🎬 Starting Evaluation Episode {episode + 1}/5")
            mujoco.mj_resetData(model, data)

            block_y = np.random.uniform(-0.20, 0.20)
            data.qpos[3:6] = [0.25, block_y, 0.43]

            print(f"🗣️ User Command: 'Reach for the red block at Y={round(block_y, 2)}'")

            for step in range(150):
                state = np.concatenate([data.qpos[:3], data.qpos[3:6]])
                state_tensor = torch.FloatTensor(state).unsqueeze(0)

                with torch.no_grad():
                    action = policy(state_tensor).squeeze(0).numpy()

                data.ctrl[:] = action
                mujoco.mj_step(model, data)

                viewer.sync()
                time.sleep(0.02)

            time.sleep(1)


if __name__ == "__main__":
    main()