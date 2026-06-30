# monoped-rl
Reinforcement learning with SAC and D4PG for monopod robot hopping, balance, and forward movement in ROS/Gazebo.

# Monoped RL — AI-Driven Rehabilitation Exoskeleton

Reinforcement learning environment for a monoped (single-leg) robot, used as a sim-to-real testbed for exoskeleton control research. Built on **ROS Noetic + Gazebo**, containerized with Docker, trained with **SAC** and **D4PG**.

**Dev machine:** HP Victus 16 | AMD Ryzen 7 7840HS | RTX 3050 6GB | 16GB RAM | Windows 11

---

## Table of Contents

- [One-Time Setup](#one-time-setup)
- [Daily Workflow](#daily-workflow)
- [Editing Code with VS Code Dev Containers](#editing-code-with-vs-code-dev-containers)
- [Reward Function Reference](#reward-function-reference)
- [Training](#training)
  - [Standing Task](#standing-task)
  - [Hopping Task (SAC)](#hopping-task-sac)
  - [Hopping Task (D4PG)](#hopping-task-d4pg)
  - [Resuming Training](#resuming-training)
- [Model Testing & Validation](#model-testing--validation)
- [ROS Topic Monitoring](#ros-topic-monitoring)
- [Troubleshooting](#troubleshooting)
- [Quick Reference](#quick-reference)

---

## One-Time Setup

These steps only need to be done once per machine.

### 1. Docker Desktop
- Verify GPU passthrough: `nvidia/cuda:11.8.0-base-ubuntu20.04`
- Confirm engine health with `hello-world`

### 2. VcXsrv (X Server for Gazebo GUI)
Gazebo is a Linux GUI app — Windows has no native X11 server, so VcXsrv provides the display target.

Download: https://sourceforge.net/projects/vcxsrv/

XLaunch config:
1. Display settings → **Multiple windows**, Display number **0**
2. Client startup → **Start no client**
3. Extra settings → enable **Clipboard**, **Primary Selection**, **Native opengl**, **Disable access control**
4. Finish, then allow VcXsrv through Windows Firewall (Private + Public)

### 3. Pull the Project Image
```bash
docker pull ntklab/monoped_rl:latest
```
Contains ROS Noetic + Gazebo + the monoped RL workspace at `/root/monoped_ws`.

### 4. VS Code Dev Containers Extension
Install **Dev Containers** (Microsoft) in VS Code — used to edit reward function files directly inside the running container.

---

## Daily Workflow

Run every time you start working on the project.

**Step 1 — Start Docker Desktop**, wait for the whale icon to stop animating.

**Step 2 — Launch XLaunch** with the same 4-screen config above. Confirm the X icon appears in the system tray.

**Step 3 — Open PowerShell** (use one terminal consistently — running `docker run` twice creates two containers).

**Step 4 — Set the display IP** (auto-detects current Wi-Fi IP):
```powershell
$DISPLAY_IP = (Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias 'Wi-Fi').IPAddress
echo $DISPLAY_IP   # should be 10.x.x.x
```

**Step 5 — Launch the container**

First time only (creates a persistent named container):
```powershell
docker run -it --gpus all -e DISPLAY=$DISPLAY_IP`:0.0 -e QT_X11_NO_MITSHM=1 --name exo_project ntklab/monoped_rl:latest bash
```

Every day after that:
```powershell
docker start -ai exo_project
```

| Flag | Purpose |
|---|---|
| `-it` | Interactive terminal |
| `--gpus all` | GPU access (RTX 3050) |
| `-e DISPLAY=...` | Points Gazebo GUI output to VcXsrv |
| `-e QT_X11_NO_MITSHM=1` | Fixes Qt/Gazebo shared memory issue on Windows |
| `--name exo_project` | Names the container for reuse |

**Step 6 — Start the simulation environment** (inside the container):
```bash
source /opt/ros/noetic/setup.sh
cd /root/monoped_ws
source devel_isolated/setup.sh
roslaunch my_legged_robots_sims main.launch
```
Gazebo should open on the Windows desktop within 20–30 seconds.

Alternative (hopper sim): `roslaunch my_hopper_training main.launch`

---

## Editing Code with VS Code Dev Containers

The container must already be running before attaching VS Code.

1. Open VS Code → **Remote Explorer** → **Dev Containers** → find `ntklab/monoped_rl:latest`
2. Click the **→** arrow to attach
3. `Ctrl+K Ctrl+O` → open folder: `/root/monoped_ws/src/my_hopper_training/src`
4. Edit `monoped_env.py` (reward function), `monoped_state.py`, `start_training_v2.py` — `Ctrl+S` saves instantly inside the container

**Find all reward-related files:**
```bash
find /root/monoped_ws -name "*.py" | xargs grep -l "reward" 2>/dev/null
```

**Disconnecting:**
- Quick: just close the VS Code window (container keeps running)
- Clean: `Ctrl+Shift+P` → `Remote: Close Remote Connection`
- Full stop: close VS Code → `Ctrl+C` in PowerShell to stop roslaunch → `exit`

**Persistence:** the container is named (`exo_project`) and launched without `--rm`, so edits persist across `docker start -ai exo_project` sessions. Don't run `docker rm exo_project` unless you've backed up:
```powershell
docker cp exo_project:/root/monoped_ws/src/my_hopper_training/src/monoped_env.py C:\Users\nitis\Desktop\monoped_env.py
```

---

## Reward Function Reference

File: `/root/monoped_ws/src/my_hopper_training/src/monoped_env.py`

| Reward Component | What It Does | To Change Standing → Jumping |
|---|---|---|
| Joint Position Reward | Keeps joints near default standing position | Reduce weight |
| Joint Effort Reward | Penalizes excessive joint torques | Reduce — jumping needs explosive torque |
| Contact Force Reward | Encourages stable ground contact | Invert or zero out — jumping means leaving ground |
| Orientation Reward | Encourages upright body posture | Keep as is or adjust slightly |
| Distance from Desired Point | Keeps robot at a target XYZ position | Change desired point to above ground level |

---

## Training

### Initial Permissions
```bash
docker exec -it exo_project bash
chmod +x /root/monoped_ws/src/my_hopper_training/src/*.py
```

Use **three terminals**: simulation, training, monitoring.

### Standing Task

**Terminal 1 — Simulation**
```bash
docker exec -it exo_project bash
source /opt/ros/noetic/setup.sh
cd /root/monoped_ws
source devel_isolated/setup.sh
roslaunch my_legged_robots_sims main.launch
```

**Terminal 2 — Training**
```bash
docker exec -it exo_project bash
source /opt/ros/noetic/setup.sh
cd /root/monoped_ws
source devel_isolated/setup.sh
cd src/my_hopper_training_sac/sr
python3 start_training_v3.py
```

### Hopping Task (SAC)
```bash
docker exec -it exo_project bash
source /opt/ros/noetic/setup.sh
cd /root/monoped_ws
source devel_isolated/setup.sh
cd src/my_hopper_training_sac/src
python3 start_training_hop.py
```

### Hopping Task (D4PG)
```bash
docker exec -it exo_project bash
source /opt/ros/noetic/setup.sh
cd /root/monoped_ws
source devel_isolated/setup.sh
cd src/my_hopper_training_d4pg/src
python3 start_training_hop.py
```

**Terminal 3 — Monitoring (TensorBoard)**
```bash
# SAC logs
python -m tensorboard.main --logdir="c:/xx/xx/"

# D4PG logs — convert CSV logs first (run convert.py in the same dir as logs, on Windows)
python convert.py
tensorboard --logdir=tensorboard --bind_all
```

### Resuming Training

**SAC — Standing model:**
```bash
docker exec -it exo_project bash
source /opt/ros/noetic/setup.sh
cd /root/monoped_ws
source devel_isolated/setup.sh
cd src/my_hopper_training_sac/src
python3 resume_training_stand.py /root/monoped_ws/src/model/monoped_run_yyyymmdd_xxxxxx/checkpoints/monoped_checkpoint_x0000_steps
```

**SAC — Hopping model:**
```bash
python3 resume_training_hop.py /root/monoped_ws/src/model/hop_run_yyyymmdd_xxxxxx/checkpoints/hop_checkpoint_80000_steps
```

**D4PG model:**
```bash
docker exec -it exo_project bash
source /opt/ros/noetic/setup.sh
cd /root/monoped_ws
source devel_isolated/setup.sh
cd src/my_hopper_training_d4pg/src
python3 resume_training_d4pg.py /root/monoped_ws/src/model/d4pg_hop_run_20260626_110552/checkpoints/d4pg_checkpoint_250000_steps.pt
```

---

## Model Testing & Validation

**D4PG model:**
```bash
docker exec -it exo_project bash
source /opt/ros/noetic/setup.sh
cd /root/monoped_ws
source devel_isolated/setup.sh
rosparam load /root/monoped_ws/src/my_hopper_training_d4pg/config/learn_params_d4pg.yaml
rosparam list | grep desired_pose
cd src/my_hopper_training_d4pg/src
python3 test_model_d4pg.py /root/monoped_ws/src/model/d4pg_hop_run_20260626_110552/models/final_model.pt 20
```

**SAC — hopping model:**
```bash
rosparam load /root/monoped_ws/src/my_hopper_training_sac/config/learn_params_hop.yaml
rosparam list | grep desired_pose
cd src/my_hopper_training_sac/src
python3 test_model_visual.py /root/monoped_ws/src/model/hop_run_20260622_214013/models/final_model.zip 10
```

**SAC — standing model:**
```bash
rosparam load /root/monoped_ws/src/my_hopper_training_sac/config/learn_params.yaml
rosparam list | grep desired_pose
cd src/my_hopper_training_sac/src
python3 test_model_visual.py /root/monoped_ws/src/model/run_20260620_132816/models/best_model.zip 10
```

---

## ROS Topic Monitoring

Real-time monitoring of robot state during training/testing.

```bash
# 1. Torso pitch angle and pitch rate
rostopic echo /monoped/imu/data | grep -E "angular_velocity:|orientation:" -A3 | head -60

# 2. Body linear velocity (forward/vertical/lateral)
rostopic echo /odom | grep -A5 "linear:" | head -40

# 3. Body position — x-displacement per episode
rostopic echo /odom | grep -A5 "position:" | head -40

# 4. Contact force magnitude — airborne vs grounded
rostopic echo /lowerleg_contactsensor_state | grep -A3 "force:" | head -40

# 5. All joint positions and velocities
rostopic echo /monoped/joint_states | grep -E "position:|velocity:" -A3 | head -60

# 6. Combined snapshot — one message from each topic
echo "=== IMU DATA ===" && rostopic echo -n1 /monoped/imu/data && \
echo -e "\n=== ODOM DATA ===" && rostopic echo -n1 /odom && \
echo -e "\n=== JOINT STATES ===" && rostopic echo -n1 /monoped/joint_states && \
echo -e "\n=== CONTACT SENSOR ===" && rostopic echo -n1 /lowerleg_contactsensor_state
```

**Inspecting training logs:**
```bash
cd /root/monoped_ws/src/model
ls -la d4pg_hop_run_*/logs/
cat d4pg_hop_run_*/logs/training_log.csv | tail -50
```

**Copying the full workspace out of the container:**
```bash
docker cp exo_project:/root/monoped_ws/src "C:\Users\nitis\Desktop\new"
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `docker: error during connect` | Docker Desktop isn't running — open it and wait for the whale icon |
| Gazebo window doesn't appear | Check VcXsrv's X icon is in the system tray; re-run the `$DISPLAY_IP` step |
| `--gpus` flag not recognized | Docker Desktop → Settings → Docker Engine → confirm `nvidia` runtime is listed |
| `roslaunch: command not found` | You skipped `source /opt/ros/noetic/setup.sh` |
| `devel_isolated/setup.sh` not found | Wrong directory — `cd /root/monoped_ws` first |
| Two containers in VS Code | You ran `docker run` twice — `docker ps`, then `docker stop <id>` the extra one |
| VS Code doesn't show the container | Container isn't running — start it first |
| Remote Explorer panel is empty | Click the refresh icon |
| File edits lost after restart | You used `--rm` by mistake — recreate with `--name exo_project` and no `--rm` |
| Different Wi-Fi IP tomorrow | Always re-run the `$DISPLAY_IP` step — it auto-detects |
| VcXsrv blocked by firewall | Windows Defender Firewall → Allow an app → VcXsrv → Private + Public |

---

## Quick Reference

**Daily startup (PowerShell):**
```powershell
# 1. Start Docker Desktop, wait for tray icon
# 2. Launch XLaunch, verify X icon in tray
# 3. Open PowerShell

$DISPLAY_IP = (Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias 'Wi-Fi').IPAddress

# First time only:
docker run -it --gpus all -e DISPLAY=$DISPLAY_IP`:0.0 -e QT_X11_NO_MITSHM=1 --name exo_project ntklab/monoped_rl:latest bash

# Every day after:
docker start -ai exo_project
```

**Inside container:**
```bash
source /opt/ros/noetic/setup.sh
cd /root/monoped_ws
source devel_isolated/setup.sh
roslaunch my_legged_robots_sims main.launch
```

**Useful one-liners:**
```bash
# Find reward-related files
find /root/monoped_ws -name "*.py" | xargs grep -l "reward" 2>/dev/null

# Check running containers
docker ps

# Stop a container
docker stop exo_project
```

| Task | Script | Model Path Example |
|---|---|---|
| Standing Training | `start_training_v3.py` | `model/monoped_run_yyyymmdd_xxxxxx/` |
| Hopping Training (SAC) | `start_training_hop.py` | `model/hop_run_yyyymmdd_xxxxxx/` |
| Hopping Training (D4PG) | `start_training_hop.py` | `model/d4pg_hop_run_yyyymmdd_xxxxxx/` |
| Resume Training — Standing | `resume_training_stand.py` | Point to checkpoint or `.zip` |
| Resume Training — Hopping | `resume_training_hop.py` | Point to checkpoint or `.zip` |
| Resume Training — D4PG | `resume_training_d4pg.py` | Point to `.pt` checkpoint |
| Test Model | `test_model_visual.py` / `test_model_d4pg.py` | Point to model file + episode count |

**VS Code — connect to container:** Remote Explorer → Other Containers → `ntklab/monoped_rl` → click → → `Ctrl+K Ctrl+O` → `/root/monoped_ws/src/my_hopper_training/src`

**VS Code — disconnect:** `Ctrl+Shift+P` → `Remote: Close Remote Connection`
