"""
MAPPO交通信号灯控制 - 完整启动脚本
自动检查环境、生成SUMO文件并开始训练
"""

import os
import sys
import subprocess

def check_sumo():
    """检查SUMO是否安装"""
    print("检查SUMO安装...")
    
    if 'SUMO_HOME' not in os.environ:
        print("❌ 错误: 未设置SUMO_HOME环境变量")
        print("\n请设置SUMO_HOME，例如:")
        print("  Windows: set SUMO_HOME=D:\\SUMO")
        print("  Linux:   export SUMO_HOME=/usr/share/sumo")
        return False
    
    # 添加SUMO tools到路径
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)
    
    # 检查sumo命令
    try:
        result = subprocess.run(['sumo', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✓ SUMO已安装: {result.stdout.split()[2]}")
            return True
    except FileNotFoundError:
        print("❌ 错误: 找不到sumo命令")
        print(f"SUMO_HOME设置为: {os.environ['SUMO_HOME']}")
        return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False
    
    return False


def check_python_packages():
    """检查Python依赖包"""
    print("\n检查Python依赖包...")
    
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'traci': 'SUMO TraCI'
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"❌ {name} 未安装")
            missing.append(package)
    
    if missing:
        print(f"\n请安装缺失的包:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True


def generate_sumo_files():
    """生成SUMO路网和路由文件"""
    print("\n生成SUMO文件...")
    
    # 检查文件是否已存在
    files_exist = all(os.path.exists(f) for f in 
                     ['simple.net.xml', 'simple.rou.xml', 'simple.sumocfg'])
    
    if files_exist:
        print("✓ SUMO文件已存在")
        user_input = input("是否重新生成? (y/N): ").strip().lower()
        if user_input != 'y':
            return True
    
    # 生成路网
    print("  生成路网文件...")
    try:
        cmd = [
            "netgenerate",
            "--grid",
            "--grid.number=3",
            "--grid.length=200",
            "--tls.guess=true",
            "--output-file=simple.net.xml",
            "--default.junctions.radius=5"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("  ✓ simple.net.xml")
        else:
            print(f"  ❌ 生成失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        return False
    
    # 生成路由文件
    print("  生成路由文件...")
    routes_xml = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="13.89"/>
    
    <!-- 南北方向流量 -->
    <flow id="flow_ns_1" type="car" begin="0" end="3600" probability="0.08" 
          from="A0B0" to="B2C2"/>
    <flow id="flow_ns_2" type="car" begin="0" end="3600" probability="0.08" 
          from="B0C0" to="C2A2"/>
    <flow id="flow_sn_1" type="car" begin="0" end="3600" probability="0.08" 
          from="B2C2" to="A0B0"/>
    <flow id="flow_sn_2" type="car" begin="0" end="3600" probability="0.08" 
          from="C2A2" to="B0C0"/>
    
    <!-- 东西方向流量 -->
    <flow id="flow_ew_1" type="car" begin="0" end="3600" probability="0.08" 
          from="A0A1" to="A2A2"/>
    <flow id="flow_ew_2" type="car" begin="0" end="3600" probability="0.08" 
          from="B0B1" to="B2B2"/>
    <flow id="flow_we_1" type="car" begin="0" end="3600" probability="0.08" 
          from="A2A2" to="A0A1"/>
    <flow id="flow_we_2" type="car" begin="0" end="3600" probability="0.08" 
          from="B2B2" to="B0B1"/>
</routes>
"""
    
    try:
        with open("simple.rou.xml", "w", encoding="utf-8") as f:
            f.write(routes_xml)
        print("  ✓ simple.rou.xml")
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        return False
    
    # 配置文件已经存在，只需确认
    if os.path.exists("simple.sumocfg"):
        print("  ✓ simple.sumocfg")
    else:
        print("  ❌ simple.sumocfg 不存在")
        return False
    
    return True


def test_sumo_config():
    """测试SUMO配置文件"""
    print("\n测试SUMO配置...")
    
    try:
        cmd = ['sumo', '-c', 'simple.sumocfg', '--duration-log.disable', 
               '--no-step-log', '--end', '10']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ SUMO配置测试成功")
            return True
        else:
            print(f"❌ SUMO配置测试失败:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def create_models_dir():
    """创建模型保存目录"""
    if not os.path.exists('models'):
        os.makedirs('models')
        print("✓ 创建models目录")
    return True


def main():
    print("="*60)
    print("MAPPO交通信号灯控制 - 环境检查与启动")
    print("="*60)
    
    # 1. 检查SUMO
    if not check_sumo():
        sys.exit(1)
    
    # 2. 检查Python包
    if not check_python_packages():
        sys.exit(1)
    
    # 3. 生成SUMO文件
    if not generate_sumo_files():
        sys.exit(1)
    
    # 4. 测试SUMO配置
    if not test_sumo_config():
        sys.exit(1)
    
    # 5. 创建模型目录
    create_models_dir()
    
    print("\n" + "="*60)
    print("✓ 环境检查完成！")
    print("="*60)
    
    # 询问是否开始训练
    print("\n选择操作:")
    print("  1. 开始训练 (默认)")
    print("  2. 测试SUMO GUI")
    print("  3. 退出")
    
    choice = input("\n请选择 (1-3): ").strip()
    
    if choice == '2':
        print("\n启动SUMO GUI...")
        subprocess.run(['sumo-gui', '-c', 'simple.sumocfg'])
    elif choice == '3':
        print("退出")
    else:
        print("\n开始训练...")
        print("="*60)
        
        # 导入并运行训练
        from mappo_traffic_control import main as train_main
        train_main()


if __name__ == "__main__":
    main()
