import os
import sys
import traci
from sumolib import checkBinary  # 自动寻找可执行文件的工具

# 1. 动态设置 SUMO_HOME (如果环境变量没配好，代码里手动补救)
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = r'D:\SUMO'

# 将 SUMO 的 tools 文件夹加入系统路径，否则 import traci 可能会失败
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if tools not in sys.path:
    sys.path.append(tools)

# 2. 自动定位 sumo-gui 的位置
# checkBinary 会根据当前系统自动找 .exe 后缀
sumo_binary = checkBinary('sumo-gui')

# 3. 指定你的配置文件绝对路径
# 建议把 .sumocfg 放在你的 Python 脚本同级目录下
sumo_cfg = r"D:\ProgramFiles\traffic_program\your_config.sumocfg"

# 4. 启动仿真
sumo_cmd = [sumo_binary, "-c", sumo_cfg]

try:
    traci.start(sumo_cmd)
    print("成功在 D:\SUMO 路径下启动仿真！")

    # 运行一小段测试一下
    for _ in range(100):
        traci.simulationStep()

    traci.close()
except Exception as e:
    print(f"启动失败，请检查 D:\SUMO\bin 目录下是否有 sumo-gui.exe。错误信息: {e}")