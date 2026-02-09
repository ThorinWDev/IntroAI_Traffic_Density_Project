import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from sumo_rl import SumoEnvironment
from stable_baselines3 import PPO  # 1. 导入 PPO 而不是 DQN

# 1. 实例化 SUMO 环境
env = SumoEnvironment(
    net_file='SUMO_FILES\\map.net.xml',
    route_file='SUMO_FILES\\map.rou.xml',
    single_agent=True,                # 依然保持单智能体合并模式
    use_gui=False,
    num_seconds=3600,
    delta_time=5,
    sumo_warnings=False               # 关闭警告提高性能
)
env = Monitor(env)

# 打印调试信息，确认控制了多少灯
print(f"当前控制的信号灯列表: {env.unwrapped.ts_ids}")
print(f"合并后的动作空间大小: {env.action_space.n}")

# 2. 定义 PPO 模型
# PPO 的超参数通常比 DQN 更稳健
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,            # PPO 常用学习率通常比 DQN 稍小
    n_steps=2048,                    # 每次更新前收集的步数
    batch_size=64,                   # 每次梯度下降的样本量
    n_epochs=10,                     # 每次更新时优化的次数
    gamma=0.99,                      # 折扣因子
    device="auto"                    # 自动选择 GPU 或 CPU
)

# 3. 开始训练
print("开始使用 PPO 训练...")
model.learn(total_timesteps=100000)

# 4. 保存模型
model.save("sumo_ppo_model")

# 5. 测试模型
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()