"""
生成简单的SUMO路网和路由文件
"""

import os
import subprocess

def generate_network():
    """生成3x3网格路网，带红绿灯"""
    print("生成路网文件...")
    cmd = [
        "netgenerate",
        "--grid",
        "--grid.number=3",
        "--grid.length=200",
        "--tls.guess=true",
        "--output-file=simple.net.xml",
        "--default.junctions.radius=5"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✓ 路网文件生成成功: simple.net.xml")
    except subprocess.CalledProcessError:
        print("✗ 路网生成失败，请确保SUMO已正确安装")
        return False
    except FileNotFoundError:
        print("✗ 找不到netgenerate命令，请确保SUMO_HOME已设置")
        return False
    
    return True


def generate_routes():
    """生成路由文件"""
    print("生成路由文件...")
    
    routes_xml = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <!-- 车辆类型定义 -->
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
    
    <!-- 对角线流量 -->
    <flow id="flow_diag_1" type="car" begin="0" end="3600" probability="0.05" 
          from="A0B0" to="C2A2"/>
    <flow id="flow_diag_2" type="car" begin="0" end="3600" probability="0.05" 
          from="C2A2" to="A0B0"/>
</routes>
"""
    
    with open("simple.rou.xml", "w", encoding="utf-8") as f:
        f.write(routes_xml)
    
    print("✓ 路由文件生成成功: simple.rou.xml")
    return True


def check_files():
    """检查生成的文件"""
    files = ["simple.net.xml", "simple.rou.xml", "simple.sumocfg"]
    all_exist = True
    
    print("\n检查文件:")
    for file in files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (未找到)")
            all_exist = False
    
    return all_exist


def main():
    print("=" * 50)
    print("SUMO路网和路由文件生成工具")
    print("=" * 50)
    print()
    
    # 生成路网
    if not generate_network():
        return
    
    # 生成路由
    if not generate_routes():
        return
    
    # 检查文件
    print()
    if check_files():
        print("\n✓ 所有文件生成成功！")
        print("\n可以运行以下命令测试:")
        print("  sumo-gui -c simple.sumocfg")
        print("\n或开始训练:")
        print("  python mappo_traffic_control.py")
    else:
        print("\n✗ 部分文件缺失")


if __name__ == "__main__":
    main()
