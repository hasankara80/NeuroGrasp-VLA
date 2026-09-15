import mujoco
import mujoco.viewer
import numpy as np
import os

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


def main():
    print("🤖 Initializing Headless Data Engine...")
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    expert_observations, expert_actions, language_commands = [], [], []
    total_episodes = 500

    print(f"🚀 Generating {total_episodes} expert trajectories...")
    for episode in range(total_episodes):
        mujoco.mj_resetData(model, data)

        block_y = np.random.uniform(-0.20, 0.20)
        data.qpos[3:6] = [0.25, block_y, 0.43]

        target_j1 = np.arctan2(block_y, 0.25)

        r_xy = np.sqrt(0.25 ** 2 + block_y ** 2)
        dz = 0.43 - 0.67
        R = np.sqrt(r_xy ** 2 + dz ** 2)

        L1 = 0.25
        L2 = 0.15

        cos_elbow = np.clip((R ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2), -1.0, 1.0)
        target_j3 = np.arccos(cos_elbow)

        alpha = np.arctan2(r_xy, dz)
        cos_shoulder = np.clip((R ** 2 + L1 ** 2 - L2 ** 2) / (2 * R * L1), -1.0, 1.0)
        beta = np.arccos(cos_shoulder)
        target_j2 = (alpha - beta)

        language_instruction = f"reach for the red block at coordinates {round(0.25, 2)}, {round(block_y, 2)}"

        for step in range(100):
            action = np.array([target_j1, target_j2, target_j3])
            data.ctrl[:] = action
            mujoco.mj_step(model, data)

            state = np.concatenate([data.qpos[:3], data.qpos[3:6]])
            expert_observations.append(state)
            expert_actions.append(action)
            language_commands.append(language_instruction)

    os.makedirs("dataset", exist_ok=True)
    np.savez("dataset/vla_expert_data.npz", states=np.array(expert_observations), actions=np.array(expert_actions),
             language=np.array(language_commands))
    print(f"✅ Scaled Dataset Saved to: dataset/vla_expert_data.npz")


if __name__ == "__main__":
    main()