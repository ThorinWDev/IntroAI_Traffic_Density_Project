"""
多智能体交通信号灯控制 - MAPPO实现
使用SUMO TraCI控制多个红绿灯，最小化车辆等待时间
"""

import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import traci
from collections import deque
import random

# ==================== 神经网络定义 ====================

class Actor(nn.Module):
    """演员网络 - 输出动作概率分布"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs


class Critic(nn.Module):
    """评论家网络 - 输出状态价值"""
    def __init__(self, global_state_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(global_state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, global_state):
        x = F.relu(self.fc1(global_state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


# ==================== MAPPO智能体 ====================

class MAPPOAgent:
    """MAPPO智能体"""
    def __init__(self, state_dim, action_dim, global_state_dim, agent_id, 
                 lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, 
                 eps_clip=0.2, K_epochs=10):
        self.agent_id = agent_id
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 演员和评论家网络
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(global_state_dim).to(self.device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 旧策略网络（用于计算重要性采样比）
        self.actor_old = Actor(state_dim, action_dim).to(self.device)
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        self.memory = []
        
    def select_action(self, state, deterministic=False):
        """选择动作"""
        state = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            action_probs = self.actor_old(state)
        
        if deterministic:
            action = torch.argmax(action_probs).item()
            return action, 1.0
        else:
            dist = Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            return action.item(), action_logprob.item()
    
    def store_transition(self, state, action, action_logprob, reward, done):
        """存储经验"""
        self.memory.append((state, action, action_logprob, reward, done))
    
    def clear_memory(self):
        """清空记忆"""
        self.memory = []

    def update(self, global_states):
        """更新策略"""
        if len(self.memory) == 0:
            return 0.0, 0.0

        states = torch.FloatTensor(np.array([m[0] for m in self.memory])).to(self.device)
        actions = torch.LongTensor(np.array([m[1] for m in self.memory])).to(self.device)
        old_logprobs = torch.FloatTensor(np.array([m[2] for m in self.memory])).to(self.device)
        rewards = [m[3] for m in self.memory]
        dones = [m[4] for m in self.memory]
        
        # 计算折扣回报
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        
        # 全局状态
        global_states_tensor = torch.from_numpy(np.array(global_states)).float().to(self.device)
        
        # 优化K轮
        actor_losses = []
        critic_losses = []
        
        for _ in range(self.K_epochs):
            # 评估当前动作
            action_probs = self.actor(states)
            dist = Categorical(action_probs)
            new_logprobs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            # 状态价值
            state_values = self.critic(global_states_tensor).squeeze()
            
            # 优势函数
            advantages = returns - state_values.detach()

            # 重要性采样比
            ratios = torch.exp(new_logprobs - old_logprobs)
            
            # PPO损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()
            
            # 评论家损失
            critic_loss = F.mse_loss(state_values, returns)
            
            # 更新演员
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        # 更新完成后清空 memory
        self.memory.clear()

        # 更新旧策略
        self.actor_old.load_state_dict(self.actor.state_dict())

        return np.mean(actor_losses), np.mean(critic_losses)


# ==================== SUMO环境 ====================

class SumoTrafficEnv:
    """SUMO交通环境"""
    def __init__(self, sumo_cfg_file, gui=False, max_steps=3600):
        self.sumo_cfg_file = sumo_cfg_file
        self.gui = gui
        self.max_steps = max_steps
        self.step_count = 0
        
        # SUMO二进制文件
        if gui:
            self.sumo_binary = "sumo-gui"
        else:
            self.sumo_binary = "sumo"
        
        self.traffic_lights = []
        self.action_duration = 10  # 每个动作持续10秒
        
        # 动作空间：每个红绿灯的相位选择
        # 假设每个红绿灯有4个相位（南北直行、南北左转、东西直行、东西左转）
        self.num_phases = 4
        
    def start_sumo(self):
        """启动SUMO"""
        sumo_cmd = [self.sumo_binary, "-c", self.sumo_cfg_file, 
                    "--no-warnings", "--no-step-log"]
        traci.start(sumo_cmd)
        
        # 获取所有交通灯ID
        self.traffic_lights = traci.trafficlight.getIDList()
        #print(f"找到 {len(self.traffic_lights)} 个交通灯: {self.traffic_lights}")
        
    def reset(self):
        """重置环境"""
        if traci.isLoaded():
            traci.close()
        
        self.start_sumo()
        self.step_count = 0
        
        # 获取初始状态
        states = self.get_states()
        global_state = self.get_global_state()
        
        return states, global_state
    
    def get_state(self, tl_id):
        """获取单个红绿灯的状态"""
        # 获取该红绿灯控制的车道
        lanes = traci.trafficlight.getControlledLanes(tl_id)
        
        # 统计每条车道的车辆数和等待时间
        queue_lengths = []
        waiting_times = []
        
        for lane in set(lanes):  # 去重
            # 车道上的车辆数
            queue_length = traci.lane.getLastStepHaltingNumber(lane)
            queue_lengths.append(queue_length)
            
            # 车道上的平均等待时间
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
            if vehicle_ids:
                waiting_time = np.mean([traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids])
            else:
                waiting_time = 0
            waiting_times.append(waiting_time)
        
        # 当前相位
        current_phase = traci.trafficlight.getPhase(tl_id)

        # 填充到固定长度（假设最多2条车道）
        max_lanes = 2
        while len(queue_lengths) < max_lanes:
            queue_lengths.append(0)
            waiting_times.append(0)

        # 构建状态向量 一维list变量
        state = queue_lengths[:max_lanes] + waiting_times[:max_lanes] + [current_phase]
        
        return np.array(state, dtype=np.float32)
    
    def get_states(self):
        """获取所有红绿灯的状态"""
        states = {}
        for tl_id in self.traffic_lights:
            states[tl_id] = self.get_state(tl_id)
        return states
    
    def get_global_state(self):
        """获取全局状态（所有红绿灯状态的拼接）"""
        states = self.get_states()
        global_state = np.concatenate([states[tl_id] for tl_id in self.traffic_lights])
        return global_state
    
    def step(self, actions):
        """执行动作"""
        # 设置每个红绿灯的相位
        for tl_id, action in actions.items():
            traci.trafficlight.setPhase(tl_id, action)
        
        # 运行SUMO模拟
        total_waiting_time = 0
        for _ in range(self.action_duration):
            traci.simulationStep()
            self.step_count += 1
            
            # 计算所有车辆的“等待罚时”,等待时间越长罚时增长率越高
            for vehicle_id in traci.vehicle.getIDList():
                total_waiting_time += traci.vehicle.getWaitingTime(vehicle_id)
        
        # 获取新状态
        next_states = self.get_states()
        next_global_state = self.get_global_state()
        
        # 计算奖励（负的等待时间）
        reward = -total_waiting_time / 100.0  # 归一化
        
        # 检查是否结束
        done = (self.step_count >= self.max_steps) or (traci.simulation.getMinExpectedNumber() <= 0)
        
        return next_states, next_global_state, reward, done
    
    def close(self):
        """关闭SUMO"""
        if traci.isLoaded():
            traci.close()


# ==================== 训练函数 ====================

def train_mappo(env, agents, num_episodes=1000, update_interval=20):
    """训练MAPPO"""
    episode_rewards = []
    global_states_history = []

    for episode in tqdm(range(num_episodes)):
        states, global_state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            actions = {}
            
            # 每个智能体选择动作
            for tl_id in env.traffic_lights:
                agent = agents[tl_id]
                state = states[tl_id]
                action, action_logprob = agent.select_action(state)
                actions[tl_id] = action
                
                # 存储部分经验（稍后添加奖励）
                agent.store_transition(state, action, action_logprob, 0, False)
            
            # 存储全局状态
            global_states_history.append(global_state)
            
            # 执行动作
            next_states, next_global_state, reward, done = env.step(actions)
            
            # 更新所有智能体的奖励
            for tl_id in env.traffic_lights:
                agent = agents[tl_id]
                # 更新最后一个转换的奖励
                if agent.memory:
                    state, action, action_logprob, _, _ = agent.memory[-1]
                    agent.memory[-1] = (state, action, action_logprob, reward, done)
            
            states = next_states
            global_state = next_global_state
            episode_reward += reward
        
        # 更新所有智能体
        if (episode + 1) % update_interval == 0:
            print(f"\n更新智能体 (Episode {episode + 1})...")
            for tl_id in env.traffic_lights:
                agent = agents[tl_id]
                actor_loss, critic_loss = agent.update(global_states_history)
                print(f"  {tl_id}: Actor Loss={actor_loss:.4f}, Critic Loss={critic_loss:.4f}")
                agent.clear_memory()
            global_states_history = []
        
        episode_rewards.append(episode_reward)
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{num_episodes}, Avg Reward (last 10): {avg_reward:.2f}")
        
        # 保存模型
        if (episode + 1) % 100 == 0:
            save_models(agents, f"models/mappo_episode_{episode + 1}")
    
    return episode_rewards


def save_models(agents, save_dir):
    """保存模型"""
    os.makedirs(save_dir, exist_ok=True)
    for tl_id, agent in agents.items():
        torch.save({
            'actor': agent.actor.state_dict(),
            'critic': agent.critic.state_dict(),
        }, f"{save_dir}/{tl_id}.pth")
    print(f"模型已保存到 {save_dir}")


def load_models(agents, load_dir):
    """加载模型"""
    for tl_id, agent in agents.items():
        checkpoint = torch.load(f"{load_dir}/{tl_id}.pth")
        agent.actor.load_state_dict(checkpoint['actor'])
        agent.actor_old.load_state_dict(checkpoint['actor'])
        agent.critic.load_state_dict(checkpoint['critic'])
    print(f"模型已从 {load_dir} 加载")


# ==================== 主程序 ====================

def main():
    # SUMO配置文件路径
    sumo_cfg_file = "simple.sumocfg"
    
    # 创建环境
    env = SumoTrafficEnv(sumo_cfg_file, gui=False, max_steps=3600)
    
    # 初始化环境以获取交通灯列表
    states, global_state = env.reset()
    
    # 状态和动作维度
    state_dim = len(states[env.traffic_lights[0]])  # 单个智能体的状态维度
    action_dim = 4  # 4个相位选择
    global_state_dim = len(global_state)  # 全局状态维度

    # 训练设备
    if torch.cuda.is_available():
        print("trained with GPU")
    else:
        print("trained with CPU")

    # 创建MAPPO智能体
    agents = {}
    for tl_id in env.traffic_lights:
        agents[tl_id] = MAPPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            global_state_dim=global_state_dim,
            agent_id=tl_id,
            lr_actor=3e-4,
            lr_critic=1e-3,
            gamma=0.99,
            eps_clip=0.2,
            K_epochs=10
        )
    
    print(f"创建了 {len(agents)} 个MAPPO智能体")
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}, 全局状态维度: {global_state_dim}")
    
    # 训练
    print("\n开始训练！")
    episode_rewards = train_mappo(env, agents, num_episodes=1000, update_interval=20)
    
    # 关闭环境
    env.close()
    
    print("\n训练完成！")


if __name__ == "__main__":
    # 确保SUMO在系统PATH中
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("请设置环境变量 'SUMO_HOME'")
    
    main()
