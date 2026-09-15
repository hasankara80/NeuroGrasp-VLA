# NeuroGrasp-VLA: Language-Conditioned Imitation Learning for Robotic Manipulation

An end-to-end Behavior Cloning (BC) and Imitation Learning pipeline built for robotic manipulation. This project demonstrates a programmatic synthetic data engine, analytical Inverse Kinematics with domain clipping, PyTorch neural policy training, and closed-loop evaluation in MuJoCo physics simulation.

<img width="800" height="948" alt="ScreenRecording2026-09-15at15 39 45-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/954a1f30-d31c-4eb4-9e52-15d57bc23839" />


## 🧠 System Architecture

* **Synthetic Data Engine (`generate_expert_data.py`):** Programmatically generates 50,000 trajectory steps across randomized spatial object positions, computing analytical Inverse Kinematics with domain-clipped Law of Cosines to prevent `NaN` physics collapses.
* **Behavior Cloning Policy (`train_bc_policy.py`):** Trains a multi-layer perceptron (MLP) in PyTorch mapping robot joint states and natural language target commands directly to continuous motor action setpoints.
* **Closed-Loop Simulation (`evaluate_vla_policy.py`):** Evaluates the neural policy in MuJoCo across randomized episodes, achieving robust spatial tracking and physical collision dynamics.

---

## 🚀 Quickstart Guide

**1. Generate Expert Dataset:**
```bash
python generate_expert_data.py
```

**2. Train the Behavior Cloning Policy:**
```bash
python train_bc_policy.py
```

**3. Evaluate in Closed-Loop Simulation:**
```bash
mjpython evaluate_vla_policy.py
```

## 🛠️ Technical Stack

Simulation Physics: MuJoCo (mujoco)

Machine Learning: PyTorch (torch.nn, torch.optim)

Data Processing: NumPy

