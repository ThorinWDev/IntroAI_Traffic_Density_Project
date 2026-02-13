"""
测试训练好的MAPPO模型
"""

import os
import sys
import numpy as np
import torch
from mappo_traffic_control import (
    SumoTrafficEnv, MAPPOAgent, load_models
)


def test_model(model_dir, num_episodes=10, gui=True):
    """测试训练好的模型"""
    
    # SUMO配置文件
    sumo_cfg_file = "simple.sumocfg"
    
    # 创建环境
    env = SumoTrafficEnv(sumo_cfg_file, gui=gui, max_steps=3600)
    
    # 初始化环境
    states, global_state = env.reset()
    
    # 创建智能体
    state_dim = len(states[env.traffic_lights[0]])
    action_dim = 4
    global_state_dim = len(global_state)
    
    agents = {}
    for tl_id in env.traffic_lights:
        agents[tl_id] = MAPPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            global_state_dim=global_state_dim,
            agent_id=tl_id
        )
    
    # 加载模型
    print(f"从 {model_dir} 加载模型...")
    load_models(agents, model_dir)
    
    # 测试多个回合
    test_rewards = []
    
    for episode in range(num_episodes):
        states, global_state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        total_waiting_time = 0
        total_queue_length = 0
        
        print(f"\n{'='*50}")
        print(f"测试回合 {episode + 1}/{num_episodes}")
        print(f"{'='*50}")
        
        while not done:
            actions = {}
            
            # 每个智能体选择动作（使用确定性策略）
            for tl_id in env.traffic_lights:
                agent = agents[tl_id]
                action, _ = agent.select_action(states[tl_id], deterministic=True)
                actions[tl_id] = action
            
            # 执行动作
            next_states, next_global_state, reward, done = env.step(actions)
            
            # 统计信息
            import traci
            current_waiting = sum(traci.vehicle.getWaitingTime(vid) 
                                 for vid in traci.vehicle.getIDList())
            current_queue = sum(traci.lane.getLastStepHaltingNumber(lane) 
                               for lane in traci.lane.getIDList())
            
            total_waiting_time += current_waiting
            total_queue_length += current_queue
            
            episode_reward += reward
            step += 1
            
            # 打印实时信息
            if step % 10 == 0:
                print(f"  步数: {step:4d} | "
                      f"奖励: {reward:8.2f} | "
                      f"等待车辆: {current_queue:4.0f} | "
                      f"总等待时间: {current_waiting:8.1f}s")
            
            states = next_states
            global_state = next_global_state
        
        test_rewards.append(episode_reward)
        
        # 打印回合总结
        avg_waiting = total_waiting_time / step
        avg_queue = total_queue_length / step
        
        print(f"\n回合总结:")
        print(f"  总奖励: {episode_reward:.2f}")
        print(f"  平均等待时间: {avg_waiting:.2f}s")
        print(f"  平均排队长度: {avg_queue:.2f}")
    
    env.close()
    
    # 打印测试结果
    print(f"\n{'='*50}")
    print("测试结果汇总")
    print(f"{'='*50}")
    print(f"平均奖励: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"最佳奖励: {np.max(test_rewards):.2f}")
    print(f"最差奖励: {np.min(test_rewards):.2f}")
    
    return test_rewards


def compare_with_fixed_time():
    """与固定时间信号灯方案对比"""
    print("\n" + "="*50)
    print("对比测试：MAPPO vs 固定时间控制")
    print("="*50)
    
    sumo_cfg_file = "simple.sumocfg"
    
    # 测试固定时间控制
    print("\n测试固定时间控制...")
    env = SumoTrafficEnv(sumo_cfg_file, gui=False, max_steps=3600)
    states, _ = env.reset()
    
    fixed_rewards = []
    for _ in range(5):
        states, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 固定时间：每个相位30秒，循环切换
            actions = {}
            for tl_id in env.traffic_lights:
                # 简单循环相位
                phase = (env.step_count // 30) % 4
                actions[tl_id] = phase
            
            states, _, reward, done = env.step(actions)
            episode_reward += reward
        
        fixed_rewards.append(episode_reward)
    
    env.close()
    
    print(f"固定时间控制平均奖励: {np.mean(fixed_rewards):.2f}")
    
    # 测试MAPPO
    print("\n测试MAPPO控制...")
    mappo_rewards = test_model("models/mappo_episode_1000", num_episodes=5, gui=False)
    
    # 对比
    print(f"\n{'='*50}")
    print("对比结果")
    print(f"{'='*50}")
    print(f"固定时间控制: {np.mean(fixed_rewards):.2f} ± {np.std(fixed_rewards):.2f}")
    print(f"MAPPO控制:    {np.mean(mappo_rewards):.2f} ± {np.std(mappo_rewards):.2f}")
    
    improvement = (np.mean(mappo_rewards) - np.mean(fixed_rewards)) / abs(np.mean(fixed_rewards)) * 100
    print(f"\n性能提升: {improvement:.1f}%")


def main():
    # 确保SUMO在PATH中
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("请设置环境变量 'SUMO_HOME'")
    
    import argparse
    parser = argparse.ArgumentParser(description='测试MAPPO交通信号灯控制模型')
    parser.add_argument('--model-dir', type=str, default='models/mappo_episode_1000',
                       help='模型目录路径')
    parser.add_argument('--episodes', type=int, default=10,
                       help='测试回合数')
    parser.add_argument('--gui', action='store_true',
                       help='使用SUMO GUI显示')
    parser.add_argument('--compare', action='store_true',
                       help='与固定时间控制对比')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_with_fixed_time()
    else:
        test_model(args.model_dir, args.episodes, args.gui)


if __name__ == "__main__":
    main()
