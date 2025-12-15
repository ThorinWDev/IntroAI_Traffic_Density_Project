import sys
import os
import cv2
import time
import random
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QFrame, QGroupBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QBrush, QPen, QFont
from ultralytics import YOLO
import pyqtgraph as pg
import torch # 确保已导入 torch

# --- 🎯 全局设备配置：启用 Apple Silicon MPS 加速 ---
if torch.backends.mps.is_available():
    GLOBAL_DEVICE = torch.device("mps")
    print(f"✅ Torch 设备配置: Apple Silicon (MPS) 加速已启用。")
elif torch.cuda.is_available():
    GLOBAL_DEVICE = torch.device("cuda")
    print(f"✅ Torch 设备配置: NVIDIA/CUDA 加速已启用。")
else:
    GLOBAL_DEVICE = torch.device("cpu")
    print(f"⚠️ Torch 设备配置: 仅使用 CPU 运行。")
# -----------------------------------------------------
# ==========================================
# 🔧 配置区域 (用户需修改此处)
# ==========================================
# 请确保这四个视频文件存在，或者修改为你的绝对路径
VIDEO_PATHS = {
    "North": "north.mp4",
    "South": "south.mp4",
    "West": "west.mp4",
    "East": "east.mp4"
}

# 交通参数配置
MIN_GREEN_TIME = 5  # 最小绿灯时间 (秒)
MAX_GREEN_TIME = 20  # 最大绿灯时间 (秒) - 超过这个时间强制变灯
Congestion_THRESHOLD = 15  # 拥堵阈值 (辆)，超过这个数视为拥堵


# ==========================================
# 🧠 第一部分：模型与算法接口 (核心大脑)
# ==========================================

# ==========================================
# 🧠 第一部分：模型与算法接口 (核心大脑)
# ==========================================

# 1. 导入 YOLO 库 (新增)
from ultralytics import YOLO


class YOLO_Interface:
    """
    这里是连接你们训练好的模型的接口。
    """

    def __init__(self):
        # 2. 加载你队友训练好的 best.pt
        # 确保 best.pt 文件在当前目录下
        print("正在加载 YOLO 模型...")
        # 🚨 关键修改：移除 device=GLOBAL_DEVICE
        # 仅加载模型。设备将在 detect 方法中设置
        self.model = YOLO('best.pt')

        # 将全局设备变量保存为类的属性，以便在 detect 中使用
        self.device = GLOBAL_DEVICE  # <--- 保留设备信息
        print("模型加载成功！")

    def detect(self, cv_image):
        """
        输入: cv_image (OpenCV 读取的每一帧图片)
        输出: (vehicle_count, annotated_image)
        """
        # 3. 进行推理 (verbose=False 防止控制台刷屏)
        results = self.model(cv_image, verbose=False)[0]

        # 4. 获取车辆数量
        # results.boxes 包含了检测到的所有框
        # 如果你们只需要统计"汽车"，可能需要过滤类别(cls)，
        # 但如果是演示，直接统计所有框通常没问题。
        vehicle_count = len(results.boxes)

        # 5. 获取画好框的图片 (YOLO自带绘图功能)
        # plot() 返回的是 BGR 格式的 numpy 数组，正好给 OpenCV 用
        annotated_image = results.plot()

        return vehicle_count, annotated_image


# ==========================================
# 🧠 算法核心部分 (请替换原有的 TrafficAlgorithm 类)
# ==========================================

class TrafficAlgorithm:
    """
    自适应交通调度算法
    """
    # --- 算法参数 (可以根据演示视频的情况微调) ---
    MIN_GREEN = 5  # 最小绿灯时长 (秒)
    MAX_GREEN = 25  # 最大绿灯时长 (秒)

    # 阈值：当车辆数超过多少时，视为“拥堵/繁忙”
    BUSY_THRESHOLD = 8
    # 阈值：当车辆数低于多少时，视为“空闲”
    EMPTY_THRESHOLD = 3

    @staticmethod
    def calculate_next_action(current_state, elapsed_time, ns_pressure, ew_pressure):
        """
        输入:
            current_state: 当前状态 (如 'NS_GREEN')
            elapsed_time: 当前状态已经运行了多少秒
            ns_pressure: 南北向总车数 (North + South)
            ew_pressure: 东西向总车数 (West + East)
        输出:
            (Action, Reason) -> ('KEEP'/'SWITCH', '决策原因文本')
        """

        # ------------------------------------------------
        # 1. 黄灯逻辑：固定时长，不做智能判断
        # ------------------------------------------------
        if 'YELLOW' in current_state:
            if elapsed_time >= 3:  # 黄灯固定3秒
                return 'SWITCH', '黄灯结束'
            else:
                return 'KEEP', '黄灯倒计时...'

        # ------------------------------------------------
        # 2. 绿灯逻辑：智能调度核心
        # ------------------------------------------------

        # 确定谁是通行方(Green)，谁是等待方(Red)
        if 'NS' in current_state:
            green_pressure = ns_pressure  # 当前通行方车数
            red_pressure = ew_pressure  # 当前等待方车数
        else:  # 'EW' in current_state
            green_pressure = ew_pressure
            red_pressure = ns_pressure

        # === 规则 A: 最小绿灯保护 ===
        if elapsed_time < TrafficAlgorithm.MIN_GREEN:
            return 'KEEP', f'最小绿灯保护 ({int(TrafficAlgorithm.MIN_GREEN - elapsed_time)}s)'

        # === 规则 B: 最大绿灯强制切换 ===
        if elapsed_time >= TrafficAlgorithm.MAX_GREEN:
            return 'SWITCH', '达到最大绿灯时长，强制切换'

        # === 规则 C: 智能续秒/截断 ===

        # C1. 空闲截断：如果当前通行方没车了 -> 马上切
        if green_pressure <= TrafficAlgorithm.EMPTY_THRESHOLD:
            return 'SWITCH', f'通行方空闲 (车数 {green_pressure} < {TrafficAlgorithm.EMPTY_THRESHOLD}) -> 提前结束'

        # C2. 拥堵续命：如果通行方很堵，且等待方不急 -> 保持绿灯
        if green_pressure > TrafficAlgorithm.BUSY_THRESHOLD:
            # 只有当等待方压力还没爆炸时，才续命
            if red_pressure < (green_pressure * 1.5):
                return 'KEEP', f'通行方繁忙 (车数 {green_pressure}) -> 智能延长绿灯'
            else:
                return 'SWITCH', f'等待方压力过大 (车数 {red_pressure}) -> 切换'

        # C3. 默认情况：如果是普通车流，让它多跑一会，直到最大时间的一半左右再看情况
        # 这里简化处理：如果没有触发上面的空闲或拥堵，就继续保持直到碰到上限
        return 'KEEP', '车流正常通行中...'

# ==========================================
# 🧵 第二部分：多线程处理 (眼睛与手脚)
# ==========================================

class VideoProcessor(QThread):
    """
    负责读取视频 + 调用模型
    """
    frame_processed = pyqtSignal(str, QImage, int)  # 信号: 方向, 图片, 车辆数

    def __init__(self, direction, video_path):
        super().__init__()
        self.direction = direction
        self.path = video_path
        self.yolo = YOLO_Interface()  # 实例化模型接口
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {self.path}")
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 视频播完循环播放
                continue

            # 1. 缩小图片加速处理
            frame = cv2.resize(frame, (640, 360))

            # 2. 调用模型检测
            count, annotated_frame = self.yolo.detect(frame)

            # 3. 转换图片格式供 PyQt 显示
            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)

            # 4. 发送信号
            self.frame_processed.emit(self.direction, qt_image, count)

            # 控制帧率 (模拟实时流)
            time.sleep(0.04)  # 约 25 FPS

    def stop(self):
        self.running = False
        self.wait()


# ==========================================
# 🎨 第三部分：自定义 UI 组件 (颜值担当)
# ==========================================

class RealTrafficLight(QWidget):
    """
    绘制拟真的红绿灯组件
    """

    def __init__(self, orientation='vertical'):
        super().__init__()
        self.orientation = orientation
        self.state = 'red'  # red, yellow, green
        if orientation == 'vertical':
            self.setFixedSize(60, 160)
        else:
            self.setFixedSize(160, 60)

    def set_color(self, color):
        self.state = color
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 画黑盒子背景
        painter.setBrush(QBrush(QColor(30, 30, 30)))
        painter.setPen(Qt.NoPen)
        rect = self.rect()
        painter.drawRoundedRect(rect, 10, 10)

        # 定义灯的位置
        if self.orientation == 'vertical':
            centers = [(30, 30), (30, 80), (30, 130)]  # 红 黄 绿
        else:
            centers = [(30, 30), (80, 30), (130, 30)]  # 红 黄 绿

        colors = {
            'red': (QColor(255, 0, 0), QColor(80, 0, 0)),  # 亮红, 暗红
            'yellow': (QColor(255, 200, 0), QColor(80, 60, 0)),
            'green': (QColor(0, 255, 0), QColor(0, 60, 0))
        }

        # 绘制三个灯
        light_order = ['red', 'yellow', 'green']
        for i, color_name in enumerate(light_order):
            cx, cy = centers[i]

            # 决定颜色：如果当前状态匹配，用亮色，否则用暗色
            if self.state == color_name:
                fill_color = colors[color_name][0]
                glow_size = 20
            else:
                fill_color = colors[color_name][1]
                glow_size = 0

            # 画灯
            painter.setBrush(QBrush(fill_color))
            painter.drawEllipse(cx - 20, cy - 20, 40, 40)


class IntersectionMap(QWidget):
    """
    绘制十字路口俯视图，并将4个红绿灯放在正确位置
    """

    def __init__(self):
        super().__init__()
        self.setFixedSize(400, 400)
        self.layout = QGridLayout(self)

        # 初始化4个方向的红绿灯
        self.lights = {
            'North': RealTrafficLight('vertical'),
            'South': RealTrafficLight('vertical'),
            'West': RealTrafficLight('horizontal'),
            'East': RealTrafficLight('horizontal')
        }

        # 布局逻辑：将灯放在十字路口的四个路口处
        # 0,1 (北)
        # 1,0 (西)  1,2 (东)
        # 2,1 (南)
        self.layout.addWidget(self.lights['North'], 0, 1, alignment=Qt.AlignBottom | Qt.AlignHCenter)
        self.layout.addWidget(self.lights['West'], 1, 0, alignment=Qt.AlignRight | Qt.AlignVCenter)
        self.layout.addWidget(self.lights['East'], 1, 2, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        self.layout.addWidget(self.lights['South'], 2, 1, alignment=Qt.AlignTop | Qt.AlignHCenter)

        # 初始状态
        self.update_lights('NS_GREEN')

    def paintEvent(self, event):
        """画十字路口的马路"""
        painter = QPainter(self)
        painter.setBrush(QBrush(QColor(50, 50, 50)))  # 深灰色马路
        painter.setPen(Qt.NoPen)

        cx, cy = 200, 200
        road_width = 120

        # 纵向路
        painter.drawRect(cx - road_width // 2, 0, road_width, 400)
        # 横向路
        painter.drawRect(0, cy - road_width // 2, 400, road_width)

        # 画斑马线 (简单的白线示意)
        painter.setPen(QPen(Qt.white, 3, Qt.DashLine))
        painter.drawLine(cx - road_width // 2, cy - road_width // 2, cx + road_width // 2, cy - road_width // 2)  # 北
        painter.drawLine(cx - road_width // 2, cy + road_width // 2, cx + road_width // 2, cy + road_width // 2)  # 南
        # ... 可以添加更多细节

    def update_lights(self, state):
        """根据全局状态更新四个灯的颜色"""
        if state == 'NS_GREEN':
            self.lights['North'].set_color('green')
            self.lights['South'].set_color('green')
            self.lights['West'].set_color('red')
            self.lights['East'].set_color('red')
        elif state == 'NS_YELLOW':
            self.lights['North'].set_color('yellow')
            self.lights['South'].set_color('yellow')
            self.lights['West'].set_color('red')
            self.lights['East'].set_color('red')
        elif state == 'EW_GREEN':
            self.lights['North'].set_color('red')
            self.lights['South'].set_color('red')
            self.lights['West'].set_color('green')
            self.lights['East'].set_color('green')
        elif state == 'EW_YELLOW':
            self.lights['North'].set_color('red')
            self.lights['South'].set_color('red')
            self.lights['West'].set_color('yellow')
            self.lights['East'].set_color('yellow')


# ==========================================
# 📈 第四部分（新增）：波形图组件
# ==========================================
class TrafficWaveform(pg.PlotWidget):
    def __init__(self):
        super().__init__()
        # 1. 整体样式设置：无边框黑色背景，模拟金融大屏
        self.setBackground('#1e1e1e')
        self.setTitle("🚦 实时交通流量趋势 (Real-time Volume)", color='#fff', size='10pt')
        self.showGrid(x=True, y=True, alpha=0.3)  # 网格线透明度
        self.setLabel('left', '车辆数', color='#888')
        self.setLabel('bottom', '时间轴 (最近60秒)', color='#888')
        self.addLegend(offset=(10, 10))  # 添加图例

        # 2. 初始化数据容器 (存储最近100个采样点)
        self.history_size = 120  # 60秒 * 2次/秒
        self.data = {
            'North': np.zeros(self.history_size),
            'South': np.zeros(self.history_size),
            'West': np.zeros(self.history_size),
            'East': np.zeros(self.history_size),
        }

        # 3. 定义线条样式 (颜色, 填充)
        # 使用霓虹配色: 红, 橙, 蓝, 青
        self.curves = {}
        configs = {
            'North': {'color': '#FF5555', 'fill': (255, 85, 85, 30)},  # 红色带透明填充
            'South': {'color': '#FFAA00', 'fill': (255, 170, 0, 30)},  # 橙色
            'West': {'color': '#00AAFF', 'fill': (0, 170, 255, 30)},  # 蓝色
            'East': {'color': '#00FFCC', 'fill': (0, 255, 204, 30)},  # 青色
        }

        for direction, cfg in configs.items():
            pen = pg.mkPen(color=cfg['color'], width=2)
            # fillLevel=0 表示填充曲线到X轴之间的区域，非常有高级感
            self.curves[direction] = self.plot(
                self.data[direction],
                name=direction,
                pen=pen,
                fillLevel=0,
                fillBrush=cfg['fill']
            )

    def update_chart(self, current_counts):
        """接收最新的一帧数据，滚动更新图表"""
        for direction, count in current_counts.items():
            # 数据左移 (滚筒效果)
            self.data[direction][:-1] = self.data[direction][1:]
            self.data[direction][-1] = count

            # 刷新线条
            self.curves[direction].setData(self.data[direction])
# ==========================================
# 🚀 第四部分：主控制台
# ==========================================

class SmartTrafficCenter(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI 智能交通自适应调度系统")
        self.resize(1280, 800)
        self.setStyleSheet("background-color: #1e1e1e; color: white;")

        # 状态变量
        self.current_state = 'NS_GREEN'
        self.state_start_time = time.time()
        self.vehicle_counts = {'North': 0, 'South': 0, 'West': 0, 'East': 0}

        self.setup_ui()
        self.start_system()

    def setup_ui(self):
        main_layout = QHBoxLayout(self)

        # --- 左侧：视频监控墙 ---
        video_group = QGroupBox("实时路况监控 (YOLO Inference)")
        video_group.setStyleSheet("QGroupBox { border: 1px solid gray; font-weight: bold; }")
        video_layout = QGridLayout(video_group)

        self.video_labels = {}
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        directions = ['North', 'South', 'West', 'East']

        for i, direction in enumerate(directions):
            # 视频容器
            container = QWidget()
            v_layout = QVBoxLayout(container)

            # 画面
            lbl = QLabel("正在连接...")
            lbl.setFixedSize(320, 180)
            lbl.setStyleSheet("background-color: black; border: 2px solid #555;")
            lbl.setScaledContents(True)
            self.video_labels[direction] = lbl

            # 数据显示
            info_lbl = QLabel(f"{direction}: 等待数据...")
            info_lbl.setFont(QFont("Arial", 12))

            v_layout.addWidget(lbl)
            v_layout.addWidget(info_lbl)

            # 存引用以便更新文本
            lbl.info_ref = info_lbl

            r, c = positions[i]
            video_layout.addWidget(container, r, c)

        main_layout.addWidget(video_group, 65)

        # --- 右侧：调度指挥中心 ---
        control_panel = QFrame()
        control_panel.setStyleSheet("background-color: #2b2b2b; border-radius: 10px;")
        control_layout = QVBoxLayout(control_panel)

        # 标题
        title = QLabel("🚦 实时调度中心")
        title.setFont(QFont("SimHei", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(title)

        # 状态显示
        self.status_label = QLabel("当前状态: 南北通行 (NS_GREEN)")
        self.status_label.setFont(QFont("SimHei", 14))
        self.status_label.setStyleSheet("color: #00ff00; margin-top: 10px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.status_label)

        self.timer_label = QLabel("当前相位运行时长: 0.0s")
        self.timer_label.setFont(QFont("Arial", 14))
        self.timer_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.timer_label)

        # 十字路口拟真图
        self.map_widget = IntersectionMap()
        control_layout.addWidget(self.map_widget, alignment=Qt.AlignCenter)

        # =========== 【新增代码开始】 ===========
        # 添加波形图
        self.waveform = TrafficWaveform()
        # 设置一个固定高度，防止它挤压地图，比如 200px
        self.waveform.setFixedHeight(200)
        control_layout.addWidget(self.waveform)
        # =========== 【新增代码结束】 ===========

        # 算法日志
        log_title = QLabel("调度算法日志:")
        control_layout.addWidget(log_title)
        self.log_label = QLabel("初始化完成...\n等待车流数据...")
        self.log_label.setStyleSheet("color: gray; font-size: 11px;")
        self.log_label.setWordWrap(True)
        control_layout.addWidget(self.log_label)

        control_layout.addStretch()
        main_layout.addWidget(control_panel, 35)

    def start_system(self):
        # 1. 启动视频处理线程
        self.threads = []
        for direction in ['North', 'South', 'West', 'East']:
            # 检查文件是否存在，否则用摄像头0顶替防止崩溃（演示用）
            path = VIDEO_PATHS[direction]
            if not os.path.exists(path):
                print(f"⚠️ 警告: 找不到 {path}，使用默认测试模式")
                # 实际部署时请注释掉下面这行，确保文件必须存在
                # path = 0

            thread = VideoProcessor(direction, path)
            thread.frame_processed.connect(self.update_video_ui)
            thread.start()
            self.threads.append(thread)

        # 2. 启动核心调度计时器 (每0.5秒执行一次决策)
        self.scheduler_timer = QTimer()
        self.scheduler_timer.timeout.connect(self.run_scheduler_logic)
        self.scheduler_timer.start(500)

    @pyqtSlot(str, QImage, int)
    def update_video_ui(self, direction, qt_img, count):
        """更新监控画面和车辆数据"""
        # 更新图片
        self.video_labels[direction].setPixmap(QPixmap.fromImage(qt_img))

        # 更新数据
        color = "red" if count > Congestion_THRESHOLD else "white"
        self.video_labels[direction].info_ref.setText(
            f"📍 {direction} | 车辆数: <span style='color:{color}; font-size:16px;'>{count}</span>"
        )

        # 更新全局数据供算法使用
        self.vehicle_counts[direction] = count

    def run_scheduler_logic(self):
        """
        ⚡️ 核心：根据车流实时调整红绿灯
        """
        elapsed = time.time() - self.state_start_time
        self.timer_label.setText(f"当前相位已运行: {elapsed:.1f} s")

        # 计算南北和东西的总压力
        ns_pressure = self.vehicle_counts['North'] + self.vehicle_counts['South']
        ew_pressure = self.vehicle_counts['West'] + self.vehicle_counts['East']

        # === 👇 修改了这里 👇 ===
        # 接收两个返回值：动作 和 原因
        action, reason = TrafficAlgorithm.calculate_next_action(
            self.current_state, elapsed, ns_pressure, ew_pressure
        )

        # 更新日志显示 (这一步对于演示非常重要，让老师知道AI在思考)
        self.log_label.setText(f"决策: {action}\n原因: {reason}\nNS压力: {ns_pressure} | EW压力: {ew_pressure}")

        # =========== 【新增代码开始】 ===========
        # 喂数据给波形图 (这是让图表动起来的关键)
        self.waveform.update_chart(self.vehicle_counts)
        # =========== 【新增代码结束】 ===========

        if action == 'SWITCH':
            self.switch_phase()
        # =======================

    def switch_phase(self):
        """执行状态切换的状态机"""
        self.state_start_time = time.time()

        if self.current_state == 'NS_GREEN':
            self.current_state = 'NS_YELLOW'
        elif self.current_state == 'NS_YELLOW':
            self.current_state = 'EW_GREEN'
        elif self.current_state == 'EW_GREEN':
            self.current_state = 'EW_YELLOW'
        elif self.current_state == 'EW_YELLOW':
            self.current_state = 'NS_GREEN'

        # 更新 UI
        self.map_widget.update_lights(self.current_state)
        self.status_label.setText(f"当前状态: {self.current_state}")

        # 变色处理
        color = "#00ff00" if "GREEN" in self.current_state else (
            "#ffff00" if "YELLOW" in self.current_state else "#ff0000")
        self.status_label.setStyleSheet(f"color: {color}; margin-top: 10px;")

    def closeEvent(self, event):
        for t in self.threads:
            t.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    pg.setConfigOptions(antialias=True)  # 开启抗锯齿，线条丝滑
    app = QApplication(sys.argv)
    window = SmartTrafficCenter()
    window.show()
    sys.exit(app.exec_())