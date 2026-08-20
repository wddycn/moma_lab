# Moma Lab

`moma_lab` 是一个独立的 Isaac Lab 外部扩展，包含 Go2W 和 Go2 速度跟踪任务。

```text
RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0
RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0
```

该 ID 为了兼容原任务名而原样保留。请不要在同一个 Python 进程中同时导入
`robot_lab.tasks` 和 `moma_lab.tasks`，否则两个工程会注册同名 Gym 环境。

## 安装

*前提要求安装好5.1版本的`isaacsim`以及2.3.2版本的`isaaclab`*<br>
`moma_lab` 可以放在任意目录，不要求位于 `robot_lab` 或 Isaac Lab 源码目录中。
先激活已经安装好 Isaac Lab 的 Python 环境，然后在 `moma_lab` 工程根目录执行：

```bash
cd /path/to/moma_lab
python -m pip install -e source/moma_lab
```

## 查看任务

```bash
python scripts/tools/list_envs.py
```

## 训练

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 \
  --headless
```

## 运行策略

工程已经按 RSL-RL 的默认日志目录结构携带原训练的最终模型：

```text
logs/rsl_rl/unitree_go2w_rough/2026-07-27_16-14-34/model_19999.pt
```

因此 `play.py` 会自动找到最新运行目录及其中最新的 `model_*.pt`。在工程根目录直接运行：

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0
```

任务参数位于
`source/moma_lab/moma_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/rough_env_cfg.py`。

Go2 的 URDF 和 DAE mesh 位于
`source/moma_lab/moma_lab/data/Robots/unitree/go2_description`，运行时不依赖
Nucleus 上的 Go2 USD。例如训练粗糙地形任务：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0 \
  --headless
```

`params/env.yaml` 和 `params/agent.yaml` 是训练时配置快照，主要用于追溯；
实际创建环境和网络时仍使用当前工程中的 Python 配置，因此修改观测维度、动作维度、
关节顺序或网络结构后，旧 checkpoint 将无法直接加载。
# 如何训练自己的机器人
- 四足可参考go2
- 轮足可参考go2w

封装训练环境可以分为五层：
```text
机器人文件 → 资产配置 → 环境配置 → PPO配置 → Gym注册
```
## 1. 整理 URDF 和 meshes
建议目录：
```text
moma_lab/source/moma_lab/moma_lab/data/Robots/my_robot/
├── urdf/
│   └── my_robot.urdf
└── meshes/
    ├── base.dae
    ├── hip.dae
    ├── thigh.dae
    ├── calf.dae
    └── foot.dae
```
URDF 中的 mesh 路径可以写成：
```xml
<mesh filename="package://my_robot/meshes/base.dae"/>
```
也可以使用相对路径：
```xml
<mesh filename="../meshes/base.dae"/>
```

关键是保证 Isaac Lab 的 URDF 转换器能找到文件。复制模型后应检查：

推荐的命名方式：
```text
FR_hip_joint
FR_thigh_joint
FR_calf_joint

FL_hip_joint
FL_thigh_joint
FL_calf_joint

RR_hip_joint
RR_thigh_joint
RR_calf_joint

RL_hip_joint
RL_thigh_joint
RL_calf_joint
```
足端 link：
```text
FR_foot
FL_foot
RR_foot
RL_foot
```
使用这种命名，可以直接复用 RobotLab 的大量正则表达式和奖励函数。

---
## 2. 定义机器人资产配置
资产配置放在：
```text
moma_lab/source/moma_lab/moma_lab/assets/unitree.py
```
也可以新建：

```text
moma_lab/assets/my_robot.py
```

基本结构如下：

```python
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg

MOMA_LAB_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../data")
)

MY_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=(
            f"{MOMA_LAB_DATA_DIR}/Robots/my_robot/"
            "urdf/my_robot.urdf"
        ),
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0,
                damping=0,
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.38),
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": 0.0,
            "F.*_thigh_joint": 0.8,
            "R.*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
```

这些数值不能永远照抄 Go2，尤其需要根据真实机器人修改：

- `pos[2]`：默认站立高度。
- `joint_pos`：默认站立姿态。
- `effort_limit`：电机持续力矩。
- `saturation_effort`：峰值力矩。
- `velocity_limit`：最大关节速度。
- `stiffness` 和 `damping`：PD 参数。

初始高度过低会穿地，过高会在初始化时掉下来。默认关节角不正确会导致机器人一生成就趴下。

---

## 3. 创建机器人环境配置
推荐复制 RobotLab Go2 的目录结构：
```text
velocity/config/quadruped/my_robot/
├── __init__.py
├── rough_env_cfg.py
└── agents/
    ├── __init__.py
    └── rsl_rl_ppo_cfg.py
```
`rough_env_cfg.py` 的核心结构：
```python
from isaaclab.utils import configclass

from moma_lab.assets.my_robot import MY_ROBOT_CFG
from moma_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)


@configclass
class MyRobotRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "base"
    foot_link_name = ".*_foot"

    joint_names = [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ]

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = MY_ROBOT_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        self.scene.height_scanner.prim_path = (
            "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        )
        self.scene.height_scanner_base.prim_path = (
            "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        )

        self.actions.joint_pos.joint_names = self.joint_names
        self.actions.joint_pos.scale = {
            ".*_hip_joint": 0.125,
            "^(?!.*_hip_joint).*": 0.25,
        }
```
然后按照机器人名称配置观测：
```python
self.observations.policy.joint_pos.params[
    "asset_cfg"
].joint_names = self.joint_names

self.observations.policy.joint_vel.params[
    "asset_cfg"
].joint_names = self.joint_names
```
以及身体和足端：
```python
self.events.randomize_rigid_body_mass_base.params[
    "asset_cfg"
].body_names = [self.base_link_name]

self.rewards.feet_air_time.params[
    "sensor_cfg"
].body_names = [self.foot_link_name]
```
### 最需要检查的三个名称
```python
base_link_name = "base"
foot_link_name = ".*_foot"
joint_names = [...]
```
它们必须和 URDF 完全匹配。URDF 如果把躯干叫作 `trunk`，就应该写：

```python
base_link_name = "trunk"
```

如果足端叫作 `FR_foot_link`，则正则可以写：

```python
foot_link_name = ".*_foot_link"
```
---

## 4. 配置奖励

第一次接入时，可以以 RobotLab Go2 的奖励为起点：

```python
self.rewards.lin_vel_z_l2.weight = -2.0
self.rewards.ang_vel_xy_l2.weight = -0.05

self.rewards.joint_torques_l2.weight = -2.5e-5
self.rewards.joint_acc_l2.weight = -2.5e-7
self.rewards.joint_pos_limits.weight = -5.0
self.rewards.joint_power.weight = -2e-5

self.rewards.action_rate_l2.weight = -0.01

self.rewards.track_lin_vel_xy_exp.weight = 3.0
self.rewards.track_ang_vel_z_exp.weight = 1.5

self.rewards.feet_air_time.weight = 0.1
self.rewards.feet_air_time_variance.weight = -1.0
self.rewards.feet_slide.weight = -0.1
self.rewards.feet_height_body.weight = -5.0
self.rewards.feet_gait.weight = 0.5
self.rewards.upward.weight = 1.0
```

但不要一开始随意改变大量权重。推荐顺序：

1. 先确认机器人可以生成并保持默认姿势。
2. 确认动作能正确驱动所有关节。
3. 使用 RobotLab Go2 原始奖励进行第一次训练。
4. 根据 TensorBoard 判断具体问题。
5. 一次只调整少数奖励项。

例如：

- 机器人不愿意前进：检查速度跟踪奖励。
- 原地乱踩：检查 command、stand-still 和步态奖励。
- 抬腿过低：调整足端高度奖励。
- 步态不协调：检查 gait 和 air-time variance。
- 经常打滑：增大 feet-slide 惩罚。
- 动作剧烈抖动：增大 action-rate、joint-acc 惩罚。
- 为省力而趴着移动：检查 upward、姿态和高度奖励。

---

## 5. 配置 PPO

创建：
```text
agents/rsl_rl_ppo_cfg.py
```
可以先使用 RobotLab Go2 的配置：
```python
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class MyRobotRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "my_robot_rough"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```
这里：
```python
max_iterations = 20000
```
是 PPO 更新迭代次数，不是物理仿真步数。

每次迭代采样数量大致是：

```text
num_envs × num_steps_per_env
```

如果使用 4096 个环境：

```text
4096 × 24 = 98,304 个样本/iteration
```

---

## 6. 注册 Gym 环境

在自定义机器人的 `__init__.py` 中：

```python
import gymnasium as gym

from . import agents


gym.register(
    id="RobotLab-Isaac-Velocity-Rough-MyRobot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rough_env_cfg:"
            "MyRobotRoughEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "MyRobotRoughPPORunnerCfg"
        ),
    },
)
```

然后必须让顶层任务包导入该模块：<br>即在tasks的__init__.py里面加入：

```python
from .manager_based.locomotion.velocity.config.quadruped import my_robot
```

缺少这一步时，文件虽然存在，但 `gym.register()` 不会执行。

另外，环境 ID 应与 `list_envs.py` 的过滤规则一致。你目前的脚本筛选 `RobotLab`，因此建议继续使用：

```text
RobotLab-Isaac-...
```

---

## 7. 推荐的验证顺序

不要一上来直接训练 20,000 iterations。

### 第一步：检查注册

```bash
python scripts/tools/list_envs.py 
```

应该能看到：

```text
RobotLab-Isaac-Velocity-Rough-MyRobot-v0
```

### 第二步：检查机器人生成

建议先用 1～2 个环境启动：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-MyRobot-v0 \
  --num_envs 2
```

### 第三步：检查关节匹配

启动日志应确认：

- 12 个受控关节。
- 12 维动作空间。
- 关节顺序与 `joint_names` 一致。
- 足端正则正好匹配四个 link。
- base 正好匹配一个 link。

### 第四步：短训练

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-MyRobot-v0 \
  --num_envs 512 \
  --headless \
  agent.max_iterations=100
```

先确认没有 NaN、CUDA 错误或奖励异常，再扩大到 4096 环境。

### 第五步：正式训练

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-MyRobot-v0 \
  --headless
```

---

## 8. HIMLoco 训练

Go2 和 Uika 均保留原来的 RSL-RL PPO 任务，并各自新增一个 HIMLoco 任务。HIMLoco 使用 6 帧
本体感知历史（当前帧加 5 帧历史）、速度估计器、16 维潜变量和特权 critic。

先做小规模冒烟训练：

```bash
python scripts/reinforcement_learning/himloco/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2-HIMLoco-v0 \
  --num_envs 16 --max_iterations 2 --headless

python scripts/reinforcement_learning/himloco/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Uika-HIMLoco-v0 \
  --num_envs 16 --max_iterations 2 --headless
```

确认没有维度、NaN 或 CUDA 错误后，移除 `--num_envs` 和 `--max_iterations` 进行正式训练。
HIMLoco 日志分别写入：

```text
logs/himloco_rsl_rl/go2_himloco/
logs/himloco_rsl_rl/uika_himloco/
```

原有任务和训练命令不变，例如：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0 --headless
```
# 中断、继续训练
### 1.中断
在训练中可以直接 ctrl+c 进行中断
### 2.导出/查看最新策略
```bash
python scripts/reinforcement_learning/himloco/play.py \
  --task RobotLab-Isaac-Velocity-Rough-Uika-HIMLoco-v0 \
  --num_envs 1 \
  --checkpoint /完整路径/model_1000.pt
```
### 3.继续训练
max_iterations = 20000-2900=17100
```bash
python scripts/reinforcement_learning/himloco/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Uika-HIMLoco-v0 \
  --resume \
  --load_run 2026-08-15_12-51-55 \
  --checkpoint model_2900.pt \
  --max_iterations 17100 \
  --headless
```


# 奖励函数分两层：

## 1. **奖励函数实现在哪里**

真正的函数代码在这里：

```bash
/home/mglf/rc/moma_lab/source/moma_lab/moma_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py
```

比如这些函数都在这个文件里：

```python
track_lin_vel_xy_exp
track_ang_vel_z_exp
feet_slide
feet_gait
joint_pos_penalty
feet_air_without_cmd
```

## 2. **uika 用哪些奖励、权重是多少**

uika 的权重配置在这里：

```bash
/home/mglf/rc/moma_lab/source/moma_lab/moma_lab/tasks/manager_based/locomotion/velocity/config/quadruped/uika/rough_env_cfg.py
```

主要看 `# ------------------------------Rewards------------------------------` 下面这一段，例如：

```python
self.rewards.track_lin_vel_xy_exp.weight = 1.2
self.rewards.track_ang_vel_z_exp.weight = 0.6
self.rewards.feet_slide.weight = -0.1
self.rewards.feet_gait.weight = 0.0
self.rewards.joint_mirror = None
```

还有一份基础默认奖励配置在：

```bash
/home/mglf/rc/moma_lab/source/moma_lab/moma_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py
```

你可以这样快速找：

```bash
rg -n "track_lin_vel_xy_exp|track_ang_vel_z_exp|feet_gait|joint_mirror|Rewards" /home/mglf/rc/moma_lab/source/moma_lab/moma_lab/tasks/manager_based/locomotion/velocity
```

一般你要调 uika，优先改 `uika/rough_env_cfg.py` 里的权重；只有要新增一种奖励计算方式时，才去改 `mdp/rewards.py`。


# UIKA + MID360 前方地形高程扫描

UIKA 的 HIMLoco 任务可以将 MID360 点云预处理后的前方局部高程图作为策略输入。训练中不直接输入
原始点云，而是使用 Isaac Lab `RayCasterCfg + GridPatternCfg` 生成固定大小的规则高程采样：

```text
仿真：规则向下射线 → 地形命中高度 → 143 维局部高程图 → HIMLoco actor
实机：MID360 点云 → 转换到 base 坐标系 → ROI 裁剪和栅格化 → 同样的 143 维输入
```

这里模拟的是“由 MID360 点云生成的地形高程图”，不是 MID360 的完整 360° 激光光束。

## 1. URDF 和资产配置

训练实际使用的 UIKA URDF 是：

```text
source/moma_lab/moma_lab/data/Robots/uika/uika_description/urdf/uika_simple_collision.urdf
```

其中需要存在固定安装的 `mid360_link`：

```xml
<link name="mid360_link"/>
<joint name="mid360_joint" type="fixed">
  <parent link="base"/>
  <child link="mid360_link"/>
  <origin xyz="0.33 0 -0.08" rpy="0 2.5 0"/>
</joint>
```

UIKA 资产配置位于：

```text
source/moma_lab/moma_lab/assets/uika.py
```

为了让固定的 `mid360_link` 在导入后仍可作为 RayCaster 挂载对象，需要保留：

```python
merge_fixed_joints=False
```

修改 URDF 后应删除或刷新对应的 URDF 转换缓存，确保 Isaac Sim 使用的是最新模型。

## 2. 配置前方高程扫描器

场景传感器定义位于：

```text
source/moma_lab/moma_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py
```

当前配置为：

```python
mid360_height_scanner = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/mid360_link",
    offset=RayCasterCfg.OffsetCfg(
        pos=(0.05, 0.0, 0.0),
    ),
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(
        resolution=0.1,
        size=(1.2, 1.0),
        direction=(0.0, 0.0, -1.0),
        ordering="xy",
    ),
    mesh_prim_paths=["/World/ground"],
    debug_vis=False,
)
```

参数含义：

- `prim_path`：扫描器随 `mid360_link` 的世界位置移动。
- `offset.pos`：相对 `mid360_link` 的传感器偏移，不是相对地面的坐标。
- `ray_alignment="yaw"`：规则网格只按挂载 link 的 yaw 旋转，不随 roll/pitch 倾斜。
- `size=(1.2, 1.0)`：矩形长 1.2 m、宽 1.0 m。
- `resolution=0.1`：采样间距 0.1 m。
- `direction=(0, 0, -1)`：当前 yaw 模式下射线沿世界坐标系竖直向下。
- `mesh_prim_paths`：当前只扫描 `/World/ground` 地形，不扫描独立动态物体。

采样点数包含矩形首尾边界：

```text
(1.2 / 0.1 + 1) × (1.0 / 0.1 + 1) = 13 × 11 = 143
```

注意，UIKA URDF 中 `mid360_link` 带有较大的 pitch。欧拉角等价表示可能使 Isaac Lab 从该 link
姿态提取出的 yaw 相差约 180°，因此 `offset.x` 的正方向不一定对应机器人 base 的前方。调整扫描位置时
必须以调试可视化结果为准。如果希望正 X 始终明确表示机器人前方，可以把虚拟高程扫描器挂在 `base`，
实机点云仍通过 TF 从 `mid360_link` 转换到 `base` 后再栅格化。

`offset.z=0` 表示射线网格与 `mid360_link` 原点处于相同高度。它不会改变矩形长宽和采样数量，
但如果某段地形高于射线起点，竖直向下的射线可能无法命中。出现高台阶采样点大量消失时，可将
虚拟射线起点抬高，例如：

```python
offset=RayCasterCfg.OffsetCfg(pos=(0.05, 0.0, 0.5))
```

这只是提高虚拟高程射线的起点，不表示真实 MID360 安装在 link 上方 0.5 m。

## 3. 在 UIKA 环境中绑定和更新传感器

UIKA 环境配置位于：

```text
source/moma_lab/moma_lab/tasks/manager_based/locomotion/velocity/config/quadruped/uika/rough_env_cfg.py
```

在 `UikaRoughEnvCfg.__post_init__()` 中配置：

```python
self.scene.mid360_height_scanner.prim_path = (
    "{ENV_REGEX_NS}/Robot/mid360_link"
)
self.scene.mid360_height_scanner.update_period = (
    self.decimation * self.sim.dt
)
```

这样扫描器会按照策略控制周期更新。若把扫描器改挂到 `base`，这里的 `prim_path` 也必须同步修改。

## 4. 将射线命中点转换成高程观测

观测函数位于：

```text
source/moma_lab/moma_lab/tasks/manager_based/locomotion/velocity/mdp/observations.py
```

`mid360_height_map()` 执行以下处理：

```text
地形命中点世界高度
→ 计算相对 base 的高度差
→ 减去 UIKA 平地站立高度 0.33 m
→ 将未命中的 NaN/Inf 转换为有限边界值
→ 裁剪到 [-0.5, 0.5]
```

当前约定是：平地接近 0，坑洼趋向正值，凸起或台阶趋向负值。实机预处理必须使用相同的符号、
裁剪范围和数组展开顺序。

## 5. 加入 HIMLoco actor observation

HIMLoco 观测配置位于：

```text
source/moma_lab/moma_lab/tasks/manager_based/locomotion/velocity/himloco_env_cfg.py
```

在 `HIMPolicyCfg` 中加入：

```python
mid360_height_map = ObsTerm(
    func=mdp.mid360_height_map,
    params={
        "sensor_cfg": SceneEntityCfg("mid360_height_scanner"),
        "clip": (-0.5, 0.5),
    },
    scale=2.0,
    clip=(-1.0, 1.0),
    noise=Unoise(n_min=-0.02, n_max=0.02),
)
```

该项位于 `HIMPolicyCfg`，所以 actor 能看到高程图；`HIMCriticCfg` 继承 policy，并继续使用原有的
特权速度、外力和较大范围地形扫描。wrapper 会自动读取新维度，不需要手动填写神经网络输入维数。

当前 UIKA 单步 policy observation 预计为 188 维：

```text
速度指令 3 + 角速度 3 + 重力投影 3
+ 关节位置 12 + 关节速度 12 + 上一步动作 12
+ MID360 高程 143
= 188
```

HIMLoco 使用当前帧加 5 帧历史，因此 actor history 输入总维数预计为：

```text
188 × 6 = 1128
```

加入高程图后旧 checkpoint 的输入层尺寸不匹配，首次训练不要使用 `--resume`。如需复用旧模型，
必须实现部分权重迁移并跳过 actor 第一层及 estimator 输入层。

## 6. 可视化扫描范围

不加载 checkpoint、只查看扫描范围：

```bash
conda activate moma_lab
cd /path/to/moma_lab

python scripts/reinforcement_learning/himloco/play.py \
  --task RobotLab-Isaac-Velocity-Rough-Uika-HIMLoco-v0 \
  --num_envs 1 \
  --debug-mid360-only
```

不要添加 `--headless`。画面中显示的是 143 条垂直射线与地形的命中点，不是完整 MID360 原始点云。

使用已有策略播放并显示扫描点：

```bash
python scripts/reinforcement_learning/himloco/play.py \
  --task RobotLab-Isaac-Velocity-Rough-Uika-HIMLoco-v0 \
  --num_envs 1 \
  --checkpoint /完整路径/model.pt \
  --debug-mid360
```

应重点检查矩形是否位于预期区域、台阶和坡面是否被正确命中，以及是否存在整片未命中点。

## 7. 训练前冒烟测试

正式长时间训练前，先运行 2 个环境和 1 次迭代：

```bash
python scripts/reinforcement_learning/himloco/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Uika-HIMLoco-v0 \
  --num_envs 2 \
  --max_iterations 1 \
  --headless
```

日志应显示：

```text
num_one_step_obs: 188
num_obs (total): 1128
num_actions: 12
```

确认没有 prim 匹配、NaN/Inf、observation 维度或 CUDA 错误后，再扩大到 1024～4096 个并行环境。

## 8. 实机 MID360 数据对齐

部署时不要直接取点云中的前 143 个点。应执行：

```text
MID360 PointCloud2
→ 根据 TF 转换到 base 坐标系
→ 裁剪与仿真矩形一致的 x/y ROI
→ 按 0.1 m 分辨率划分 13 × 11 栅格
→ 每格取地面点 z 的中位数或稳健百分位
→ 使用与 mid360_height_map() 一致的相对高度和符号
→ clip 到 [-0.5, 0.5]
→ 乘 2.0
→ 严格按照 ordering="xy" 的顺序展平
→ 拼入策略单步 observation
```

空栅格不能保留 NaN。第一版可以填固定边界值或邻域插值，后续可增加 validity mask、随机坏点、
1～2 帧延迟和外参扰动，提高 sim-to-real 鲁棒性。
