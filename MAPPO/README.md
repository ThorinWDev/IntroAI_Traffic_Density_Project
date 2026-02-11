# MAPPO交通信号灯控制 - 使用说明

## 概述
这是一个使用MAPPO（Multi-Agent Proximal Policy Optimization）算法训练多个交通信号灯的强化学习项目。目标是最小化车辆的总等待时间。

## 环境要求

### 1. 安装SUMO
访问下载页面：前往 SUMO 官方下载页。

下载 MSI 安装包：在 Windows 部分，点击下载 sumo-win64-x.x.x.msi（x.x.x 为版本号）。

提示：如果你需要额外的开发功能，可以选择带 extra 后缀的版本。

运行安装程序：双击下载的 .msi 文件，按照提示点击 "Next" 完成安装。

自动配置：安装程序会自动将 SUMO 添加到系统路径中，并设置 SUMO_HOME 环境变量（这是运行 SUMO 脚本的关键）。
```bash
# Ubuntu/Linux
sudo apt-get install sumo sumo-tools sumo-doc

# 或从源码安装：https://sumo.dlr.de/docs/Installing/index.html
```

### 2. 设置环境变量
```bash
export SUMO_HOME="/usr/share/sumo"  # 根据你的安装路径调整
```

### 3. 安装Python依赖
```bash
pip install torch numpy traci
```

## 项目结构

```
.
├── mappo_traffic_control.py    # 主训练代码
├── your_network.sumocfg        # SUMO配置文件（需要自己创建）
├── your_network.net.xml        # SUMO路网文件
├── your_network.rou.xml        # SUMO路由文件
└── models/                     # 保存的模型目录
```

## SUMO配置文件准备

### 1. 创建简单的路网配置 (simple.sumocfg)
```xml
<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
               xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="simple.net.xml"/>
        <route-files value="simple.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
</configuration>
```

### 2. 创建路网文件 (simple.net.xml)
可以使用SUMO的netedit工具创建，或使用netgenerate生成：

```bash
# 生成网格路网
netgenerate --grid --grid.number=3 --grid.length=200 \
            --tls.guess=true --output-file=simple.net.xml
```

### 3. 创建路由文件 (simple.rou.xml)
```xml
<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="50"/>
    
    <flow id="flow_ns" type="car" begin="0" end="3600" probability="0.1" 
          from="edge1" to="edge2"/>
    <flow id="flow_ew" type="car" begin="0" end="3600" probability="0.1" 
          from="edge3" to="edge4"/>
</routes>
```

## 代码关键组件说明

### 1. Actor网络
- 为每个交通灯独立训练
- 输入：局部状态（队列长度、等待时间、当前相位）
- 输出：动作概率分布（选择哪个相位）

### 2. Critic网络
- 共享的价值网络
- 输入：全局状态（所有交通灯的状态拼接）
- 输出：状态价值估计

### 3. 状态表示
每个交通灯的状态包括：
- 每条车道的排队车辆数
- 每条车道的平均等待时间
- 当前信号相位

### 4. 动作空间
每个交通灯可以选择4个相位之一：
- 相位0：南北直行
- 相位1：南北左转
- 相位2：东西直行
- 相位3：东西左转

### 5. 奖励函数
```python
reward = -总等待罚时 / 100.0
```
目标是最小化所有车辆的总等待罚时，若一辆车等待了5秒，罚时为1+2+3+4+5=15秒。

## 使用方法

### 1. 修改配置
在 `main()` 函数中修改SUMO配置文件路径：
```python
sumo_cfg_file = "simple.sumocfg"  # 你的配置文件路径
```

### 2. 运行训练
```bash
python mappo_traffic_control.py
```

### 3. 可视化训练（使用GUI）
修改环境创建部分：
```python
env = SumoTrafficEnv(sumo_cfg_file, gui=True, max_steps=3600)
```

### 4. 测试训练好的模型
```python
# 加载模型
load_models(agents, "models/mappo_episode_1000")

# 测试
states, global_state = env.reset()
done = False
while not done:
    actions = {}
    for tl_id in env.traffic_lights:
        agent = agents[tl_id]
        action, _ = agent.select_action(states[tl_id], deterministic=True)
        actions[tl_id] = action
    
    states, global_state, reward, done = env.step(actions)
    print(f"Reward: {reward:.2f}")
```

## 超参数调整

### 学习率
```python
lr_actor=3e-4,    # Actor学习率
lr_critic=1e-3,   # Critic学习率
```

### PPO参数
```python
gamma=0.99,       # 折扣因子
eps_clip=0.2,     # PPO裁剪参数
K_epochs=10       # 每次更新的迭代次数
```

### 训练参数
```python
num_episodes=1000,      # 训练回合数
update_interval=20,     # 每20回合更新一次
```

## 性能优化建议

### 1. 状态表示优化
- 添加邻近交通灯的信息
- 包含车辆速度信息
- 使用更复杂的特征工程

### 2. 奖励函数改进
```python
# 多目标奖励
reward = -α * waiting_time - β * queue_length + γ * throughput
```

### 3. 网络结构
- 使用GRU/LSTM处理时序信息
- 使用注意力机制处理多车道信息
- 增加网络深度和宽度

### 4. 探索策略
- 添加entropy bonus鼓励探索
- 使用ε-greedy策略
- 实现好奇心驱动探索

## 常见问题

### Q1: TraCI连接失败
**A**: 确保SUMO_HOME环境变量设置正确，并且SUMO在PATH中。

### Q2: 训练不稳定
**A**: 尝试降低学习率，增加batch size，或使用更小的eps_clip值。

### Q3: 奖励一直为负
**A**: 这是正常的，因为我们使用负的等待时间作为奖励。关注奖励的增长趋势。

### Q4: 内存溢出
**A**: 减少update_interval或使用经验回放缓冲区限制大小。

## 扩展方向

1. **通信机制**：添加智能体间通信
2. **中心化训练**：使用中心化的Critic
3. **图神经网络**：使用GNN建模交通网络
4. **迁移学习**：在不同路网间迁移学习
5. **实时优化**：支持动态交通流

## 参考资料

- MAPPO论文: https://arxiv.org/abs/2103.01955
- SUMO文档: https://sumo.dlr.de/docs/
- TraCI API: https://sumo.dlr.de/docs/TraCI.html

## License
MIT
