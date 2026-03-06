For personal reference:

- Setting up .usd for training:
  - Get urdf. Import into isaacsim. Set joint properties. Save as .usd.
  - Link it in isaacsim_assets kinova.py 

- Training
  - activate .venv and go to IsaacLab folder
  - `./isaaclab.sh -p ../kinova_isaaclab_sim2real/scripts/rl_games/train.py --task Gen3-Reach-v0`
  
- Running on kinova
  - training gives .pth checkpoint in IsaacLab logs which has to be converted to .pt policy files using `export_policy.py` file in rl_games/
  - put this .pt file, env.yaml and agent.yaml (from log folder) to /pretrained_models/reach folder
  - clean env.yaml file using clean_env_yaml.py script in sim2real/
  - first run kinova using `ros2 launch kortex_bringup kortex_sim_control.launch.py   use_sim_time:=true   launch_rviz:=true vision:=true dof:=6` 
  - then run script `python3 run_task_reach.py` in sim2real/


# 🦾 Kinova Gen3 RL & Sim2Real Toolkit

## Overview
This repository provides a modular framework for training reinforcement learning (RL) agents on the **Kinova Gen3** robot using **Isaac Lab**, and deploying trained models either in **Isaac Sim**, or on the **real robot** via ROS2.

Built as a standalone Isaac Lab extension, it allows isolated development.

> [!IMPORTANT]
> The Sim2Real pipeline runs entirely with **ROS2** and does **not require Isaac Lab or Isaac Sim** to be installed. And for convenience, **pre-trained models are provided** so you can get started immediately without training.


| Section | What you’ll find |
|---------|------------------|
| **🛠️ Installation** | How to install Isaac Lab, clone this repo, and install the Python package. |
| **🚀 Training & Basic Testing** | Commands to train a reach-task policy with `rsl_rl` or `rl_games` and replay it in Isaac Lab. |
| **🤖 Sim-to-Real Deployment (ROS 2)** | Step-by-step instructions to test on fake hardware, then execute the exact same policy on the physical Kinova Gen3. |
| **🌟 Acknowledgements** | Credits to Isaac Lab, Kinova, community help, and INIT Lab. |
| **📚 Citation** | BibTeX citation |

https://github.com/user-attachments/assets/91d013ce-9e2a-422d-8fd5-e08fdd5c9282

## 🛠️ Installation

1. Install Isaac Lab by following the official [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) (conda recommended).  
2. Clone this repo **outside** the `IsaacLab` directory.  
3. Install from the repository:

```bash
python -m pip install -e source/gen3
```

## 🚀 Training & Basic Testing

You can train a policy on the Kinova Gen3 **Reach Task** using either `rsl_rl` or `rl_games` library:

```bash
python scripts/rsl_rl/train.py --task Gen3-Reach-v0
```

After training, a quick way to validate the behavior is to use `play.py`:

```bash
python scripts/rsl_rl/play.py --task Gen3-Reach-v0
```

This helps confirm that the learned policy performs as expected in **Isaac Lab** before attempting transfer.

## 🤖 Sim2Real Deployment (ROS2)

The Sim2Real pipeline focuses on deploying trained reinforcement learning policies directly onto the **real Kinova Gen3 robot** using a minimal **ROS2-based interface**, with no dependency on Isaac Lab or Isaac Sim at runtime.

### Fake-hardware test

You can first try to simulate the Kinova Gen3 in ROS2 using fake hardware mode:

```bash
ros2 launch kortex_bringup gen3.launch.py robot_ip:=yyy.yyy.yyy.yyy use_fake_hardware:=true
```

To test movement commands, send a simple joint trajectory:

```bash
ros2 topic pub /joint_trajectory_controller/joint_trajectory trajectory_msgs/JointTrajectory "{
  joint_names: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7],
  points: [
    { positions: [0, 0, 0, 0, 0, 0, 0], time_from_start: { sec: 1 } },
  ]
}" -1
```

Run the Reach Task with the trained policy to go to a certain predefined position:

```bash
python3 scripts/sim2real/run_task_reach.py
```

### Real robot test

The next step is to connect the same interface to the **real Kinova Gen3** and execute the learned Reach Task in real-world conditions — using the exact same model and runtime logic validated in simulation.

For that you'll run:

```bash
ros2 launch kortex_bringup gen3_lite.launch.py robot_ip:=192.168.1.10
```

And in the same way as simulation, in another terminal:

```bash
python3 scripts/sim2real/run_task_reach.py
```

Now your robot should be alternating between three positions.

## 🌟 Acknowledgements

* Isaac Lab team & contributors
  * [Isaac Lab repository](https://github.com/isaac-sim/IsaacLab)
  * [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/main/index.html)
* Johnson Sun
  * [GitHub profile](https://github.com/j3soon)
  * [UR10 Reacher RL sim2real Isaac Gym env repository](https://github.com/j3soon/OmniIsaacGymEnvs-UR10Reacher) 
* Kinova Robotics
  * [ros2_kortex repository](https://github.com/Kinovarobotics/ros2_kortex)
* INIT Lab
  * [Website](https://initrobots.ca/)
  * David St-Onge
  * Augustin Nguon

## 📚 Citation

If you use this work, please cite it as:

```bibtex
@software{Le_Lay_Kinova_Gen3_RL_2025,
  author = {Le Lay, Louis},
  doi = {https://doi.org/10.5281/zenodo.16530856},
  license = {MIT},
  month = jul,
  title = {Kinova Gen3 RL \& Sim2Real Toolkit},
  url = {https://github.com/louislelay/kinova_isaaclab_sim2real},
  version = {v1.0.0},
  year = {2025}
}
