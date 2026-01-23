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
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, pyqtSlot, QUrl
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QBrush, QPen, QFont
from PyQt5.QtMultimedia import QSoundEffect
from ultralytics import YOLO
import pyqtgraph as pg
import torch
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QFrame, QGroupBox,
    QPushButton  # 新增
)

if torch.backends.mps.is_available():
    GLOBAL_DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    GLOBAL_DEVICE = torch.device("cuda")
else:
    GLOBAL_DEVICE = torch.device("cpu")

# ==========================================
# 🔧 配置区域
# ==========================================
VIDEO_PATHS = {
    "North": "res\\north.mp4",
    "South": "res\\south.mp4",
    "West": "res\\west.mp4",
    "East": "res\\east.mp4"
}

# 交通参数配置
MIN_GREEN_TIME = 5
MAX_GREEN_TIME = 20
Congestion_THRESHOLD = 15


# ==========================================
#  第一部分：模型与算法接口 (核心大脑)
# ==========================================

class YOLO_Interface:
    """
    双模型接口：同时负责车辆计数和特种车辆检测
    """

    def __init__(self):
        self.device = GLOBAL_DEVICE

        print("正在加载 YOLO 模型 (车辆检测)...")
        self.model_cars = YOLO('pts\\ordinary.pt')  # 你的车辆模型

        print("正在加载 YOLO 模型 (救护车检测)...")
        # [新增] 加载第二个模型
        self.model_ambulance = YOLO('pts\\specific.pt')

        print("模型加载成功！")

    def detect(self, cv_image):
        """
        输入: cv_image
        输出: (vehicle_count, annotated_image, is_ambulance_detected)
        """
        # --- 1. 车辆检测 (用于计数) ---
        results_cars = self.model_cars(cv_image, verbose=False)[0]
        vehicle_count = len(results_cars.boxes)
        # 使用车辆模型的绘图结果作为基础底图
        annotated_image = results_cars.plot()

        # --- 2. 救护车检测 (用于特权) ---
        # [新增] 运行第二个模型
        results_amb = self.model_ambulance(cv_image, verbose=False, conf=0.9)[0]

        is_ambulance = False

        # [新增] 如果检测到救护车
        if len(results_amb.boxes) > 0:
            is_ambulance = True
            # 这里我们不调用 plot() 画框，而是手动添加强烈的视觉提示
            # 在图片上加一个半透明红色遮罩或者大字
            overlay = annotated_image.copy()
            cv2.rectangle(overlay, (0, 0), (overlay.shape[1], overlay.shape[0]), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, annotated_image, 0.7, 0, annotated_image)

            # 绘制醒目文字
            cv2.putText(annotated_image, "!!! AMBULANCE !!!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

            # 如果你想画框但只要框不要标签，可以手动画 results_amb.boxes 的坐标
            # 这里为了简洁，仅用全屏警告代替 bounding box

        return vehicle_count, annotated_image, is_ambulance


# ==========================================
#  算法核心部分
# ==========================================

class TrafficAlgorithm:
    MIN_GREEN = 5
    MAX_GREEN = 25
    BUSY_THRESHOLD = 8
    EMPTY_THRESHOLD = 3

    @staticmethod
    def calculate_next_action(current_state, elapsed_time, ns_pressure, ew_pressure):
        # ... (原有的普通逻辑保持不变) ...
        if 'YELLOW' in current_state:
            if elapsed_time >= 3:
                return 'SWITCH', '黄灯结束'
            else:
                return 'KEEP', '黄灯倒计时...'

        if 'NS' in current_state:
            green_pressure = ns_pressure
            red_pressure = ew_pressure
        else:
            green_pressure = ew_pressure
            red_pressure = ns_pressure

        if elapsed_time < TrafficAlgorithm.MIN_GREEN:
            return 'KEEP', f'最小绿灯保护 ({int(TrafficAlgorithm.MIN_GREEN - elapsed_time)}s)'

        if elapsed_time >= TrafficAlgorithm.MAX_GREEN:
            return 'SWITCH', '达到最大绿灯时长，强制切换'

        if green_pressure <= TrafficAlgorithm.EMPTY_THRESHOLD:
            return 'SWITCH', f'通行方空闲 ({green_pressure}) -> 提前结束'

        if green_pressure > TrafficAlgorithm.BUSY_THRESHOLD:
            if red_pressure < (green_pressure * 1.5):
                return 'KEEP', f'通行方繁忙 -> 智能延长'
            else:
                return 'SWITCH', f'等待方压力过大 -> 切换'

        return 'KEEP', '车流正常通行中...'


# ==========================================
#  第二部分：多线程处理
# ==========================================

class VideoProcessor(QThread):
    # [修改] 信号增加了一个 bool 参数：is_ambulance
    frame_processed = pyqtSignal(str, QImage, int, bool)

    def __init__(self, direction, video_path):
        super().__init__()
        self.direction = direction
        self.path = video_path
        self.yolo = YOLO_Interface()
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame = cv2.resize(frame, (640, 360))

            # [修改] 获取三个返回值
            count, annotated_frame, is_amb = self.yolo.detect(frame)

            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)

            # [修改] 发送信号
            self.frame_processed.emit(self.direction, qt_image, count, is_amb)

            time.sleep(0.04)

    def stop(self):
        self.running = False
        self.wait()


# ==========================================
#  第三部分：自定义 UI 组件 (保持不变)
# ==========================================
# ... (RealTrafficLight, IntersectionMap, TrafficWaveform 保持原样，无需修改) ...

class RealTrafficLight(QWidget):
    def __init__(self, orientation='vertical'):
        super().__init__()
        self.orientation = orientation
        self.state = 'red'
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
        painter.setBrush(QBrush(QColor(30, 30, 30)))
        painter.setPen(Qt.NoPen)
        rect = self.rect()
        painter.drawRoundedRect(rect, 10, 10)
        if self.orientation == 'vertical':
            centers = [(30, 30), (30, 80), (30, 130)]
        else:
            centers = [(30, 30), (80, 30), (130, 30)]
        colors = {'red': (QColor(255, 0, 0), QColor(80, 0, 0)), 'yellow': (QColor(255, 200, 0), QColor(80, 60, 0)),
                  'green': (QColor(0, 255, 0), QColor(0, 60, 0))}
        light_order = ['red', 'yellow', 'green']
        for i, color_name in enumerate(light_order):
            cx, cy = centers[i]
            if self.state == color_name:
                fill_color = colors[color_name][0]
            else:
                fill_color = colors[color_name][1]
            painter.setBrush(QBrush(fill_color))
            painter.drawEllipse(cx - 20, cy - 20, 40, 40)


class IntersectionMap(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(400, 400)
        self.layout = QGridLayout(self)
        self.lights = {'North': RealTrafficLight('vertical'), 'South': RealTrafficLight('vertical'),
                       'West': RealTrafficLight('horizontal'), 'East': RealTrafficLight('horizontal')}
        self.layout.addWidget(self.lights['North'], 0, 1, alignment=Qt.AlignBottom | Qt.AlignHCenter)
        self.layout.addWidget(self.lights['West'], 1, 0, alignment=Qt.AlignRight | Qt.AlignVCenter)
        self.layout.addWidget(self.lights['East'], 1, 2, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        self.layout.addWidget(self.lights['South'], 2, 1, alignment=Qt.AlignTop | Qt.AlignHCenter)
        self.update_lights('NS_GREEN')

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(QBrush(QColor(50, 50, 50)))
        painter.setPen(Qt.NoPen)
        cx, cy = 200, 200
        road_width = 120
        painter.drawRect(cx - road_width // 2, 0, road_width, 400)
        painter.drawRect(0, cy - road_width // 2, 400, road_width)
        painter.setPen(QPen(Qt.white, 3, Qt.DashLine))
        painter.drawLine(cx - road_width // 2, cy - road_width // 2, cx + road_width // 2, cy - road_width // 2)
        painter.drawLine(cx - road_width // 2, cy + road_width // 2, cx + road_width // 2, cy + road_width // 2)

    def update_lights(self, state):
        if state == 'ALL_RED':
            for light in self.lights.values():
                light.set_color('red')
            return  # 直接返回，不执行后面的逻辑
        if state == 'NS_GREEN':
            self.lights['North'].set_color('green');
            self.lights['South'].set_color('green');
            self.lights['West'].set_color('red');
            self.lights['East'].set_color('red')
        elif state == 'NS_YELLOW':
            self.lights['North'].set_color('yellow');
            self.lights['South'].set_color('yellow');
            self.lights['West'].set_color('red');
            self.lights['East'].set_color('red')
        elif state == 'EW_GREEN':
            self.lights['North'].set_color('red');
            self.lights['South'].set_color('red');
            self.lights['West'].set_color('green');
            self.lights['East'].set_color('green')
        elif state == 'EW_YELLOW':
            self.lights['North'].set_color('red');
            self.lights['South'].set_color('red');
            self.lights['West'].set_color('yellow');
            self.lights['East'].set_color('yellow')


class TrafficWaveform(pg.PlotWidget):
    def __init__(self):
        super().__init__()
        self.setBackground('#1e1e1e')
        self.setTitle("🚦 实时交通流量趋势", color='#fff', size='10pt')
        self.showGrid(x=True, y=True, alpha=0.3)
        self.addLegend(offset=(10, 10))
        self.history_size = 120
        self.data = {'North': np.zeros(self.history_size), 'South': np.zeros(self.history_size),
                     'West': np.zeros(self.history_size), 'East': np.zeros(self.history_size)}
        self.curves = {}
        configs = {'North': {'color': '#FF5555', 'fill': (255, 85, 85, 30)},
                   'South': {'color': '#FFAA00', 'fill': (255, 170, 0, 30)},
                   'West': {'color': '#00AAFF', 'fill': (0, 170, 255, 30)},
                   'East': {'color': '#00FFCC', 'fill': (0, 255, 204, 30)}}
        for direction, cfg in configs.items():
            pen = pg.mkPen(color=cfg['color'], width=2)
            self.curves[direction] = self.plot(self.data[direction], name=direction, pen=pen, fillLevel=0,
                                               fillBrush=cfg['fill'])

    def update_chart(self, current_counts):
        for direction, count in current_counts.items():
            self.data[direction][:-1] = self.data[direction][1:]
            self.data[direction][-1] = count
            self.curves[direction].setData(self.data[direction])


# ==========================================
#  第四部分：主控制台 (逻辑更新)
# ==========================================

class SmartTrafficCenter(QWidget):
    # --- [添加 1] 在类定义的最上方定义信号 ---
    switch_to_track_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI 智能交通自适应调度系统 (含特种车辆优先)")
        self.resize(1280, 800)
        self.setStyleSheet("background-color: #1e1e1e; color: white;")

        self.current_state = 'NS_GREEN'
        self.state_start_time = time.time()
        self.vehicle_counts = {'North': 0, 'South': 0, 'West': 0, 'East': 0}

        # [新增] 救护车状态跟踪
        self.ambulance_status = {'North': False, 'South': False, 'West': False, 'East': False}
        self.is_emergency_mode = False

        # 新增变量：紧急模式的锁定结束时间，确保绿灯持续到这个时间
        self.emergency_mode_end_lock_time = 0

        # [新增] 视觉闪烁定时器
        self.flash_timer = QTimer()
        self.flash_timer.timeout.connect(self.toggle_flash_effect)
        self.flash_state = False

        # [新增] 音效初始化
        self.siren_sound = QSoundEffect()
        # 请确保目录有 siren.wav，如果没有也不会报错
        self.siren_sound.setSource(QUrl.fromLocalFile("siren.wav"))
        self.siren_sound.setLoopCount(QSoundEffect.Infinite)  # 循环播放
        self.siren_sound.setVolume(0.5)

        self.setup_ui()
        self.start_system()

    def setup_ui(self):
        # ... (UI 构建代码与原版基本一致) ...
        main_layout = QHBoxLayout(self)
        video_group = QGroupBox("实时路况监控")
        video_group.setStyleSheet("QGroupBox { border: 1px solid gray; font-weight: bold; }")
        video_layout = QGridLayout(video_group)
        self.video_labels = {}
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        directions = ['North', 'South', 'West', 'East']
        for i, direction in enumerate(directions):
            container = QWidget()
            v_layout = QVBoxLayout(container)
            lbl = QLabel("正在连接...")
            lbl.setFixedSize(320, 180)
            lbl.setStyleSheet("background-color: black; border: 2px solid #555;")
            lbl.setScaledContents(True)
            self.video_labels[direction] = lbl
            info_lbl = QLabel(f"{direction}: 等待数据...")
            info_lbl.setFont(QFont("Arial", 12))
            v_layout.addWidget(lbl)
            v_layout.addWidget(info_lbl)
            lbl.info_ref = info_lbl
            r, c = positions[i]
            video_layout.addWidget(container, r, c)
        main_layout.addWidget(video_group, 65)

        control_panel = QFrame()
        control_panel.setStyleSheet("background-color: #2b2b2b; border-radius: 10px;")
        control_layout = QVBoxLayout(control_panel)

        # [修改] 给标题一个引用，方便变色
        self.title_label = QLabel("🚦 实时调度中心")
        self.title_label.setFont(QFont("SimHei", 20, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.title_label)
        # --- [添加 2] 在控制面板里塞一个跳转按钮 ---
        self.btn_track = QPushButton("🔍 进入车辆追踪模式")
        self.btn_track.setStyleSheet("""
                    QPushButton {
                        background-color: #0078d7; color: white; 
                        padding: 10px; font-weight: bold; border-radius: 5px;
                        margin-top: 10px;
                    }
                    QPushButton:hover { background-color: #005a9e; }
                """)
        # 点击按钮时，发射信号
        self.btn_track.clicked.connect(self.switch_to_track_signal.emit)
        control_layout.addWidget(self.btn_track)  # 将按钮加到控制面板布局中

        self.status_label = QLabel("当前状态: NS_GREEN")
        self.status_label.setFont(QFont("SimHei", 14))
        self.status_label.setStyleSheet("color: #00ff00; margin-top: 10px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.status_label)

        self.timer_label = QLabel("0.0s")
        self.timer_label.setFont(QFont("Arial", 14))
        self.timer_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.timer_label)

        self.map_widget = IntersectionMap()
        control_layout.addWidget(self.map_widget, alignment=Qt.AlignCenter)

        self.waveform = TrafficWaveform()
        self.waveform.setFixedHeight(200)
        control_layout.addWidget(self.waveform)

        log_title = QLabel("调度算法日志:")
        control_layout.addWidget(log_title)
        self.log_label = QLabel("...")
        self.log_label.setStyleSheet("color: gray; font-size: 11px;")
        self.log_label.setWordWrap(True)
        control_layout.addWidget(self.log_label)

        control_layout.addStretch()
        main_layout.addWidget(control_panel, 35)

    def start_system(self):
        self.threads = []
        for direction in ['North', 'South', 'West', 'East']:
            path = VIDEO_PATHS[direction]
            if not os.path.exists(path):
                pass  # path = 0 # 演示时根据需要开启摄像头
            thread = VideoProcessor(direction, path)
            thread.frame_processed.connect(self.update_video_ui)
            thread.start()
            self.threads.append(thread)

        self.scheduler_timer = QTimer()
        self.scheduler_timer.timeout.connect(self.run_scheduler_logic)
        self.scheduler_timer.start(500)

    # [修改] 槽函数接收 is_amb
    @pyqtSlot(str, QImage, int, bool)
    def update_video_ui(self, direction, qt_img, count, is_amb):
        self.video_labels[direction].setPixmap(QPixmap.fromImage(qt_img))

        # 保存全局救护车状态
        self.ambulance_status[direction] = is_amb
        self.vehicle_counts[direction] = count

        # UI 文本显示
        if is_amb:
            status_text = " AMBULANCE "
            color = "#FF0000"  # 亮红
            self.video_labels[direction].setStyleSheet("border: 4px solid red;")
        else:
            status_text = str(count)
            color = "red" if count > Congestion_THRESHOLD else "white"
            self.video_labels[direction].setStyleSheet("border: 2px solid #555;")

        self.video_labels[direction].info_ref.setText(
            f" {direction} | <span style='color:{color}; font-weight:bold;'>{status_text}</span>"
        )

    def run_scheduler_logic(self):
        """
        ⚡️ 核心：调度逻辑 (含紧急优先)
        """
        elapsed = time.time() - self.state_start_time
        self.timer_label.setText(f"相位时长: {elapsed:.1f} s")

        # ======================================
        #  紧急优先逻辑 (Override Logic)
        # ======================================
        emergency_direction = None
        for direction, is_here in self.ambulance_status.items():
            if is_here:
                emergency_direction = direction
                break

        # 如果检测到救护车
        if emergency_direction:
            self.activate_emergency_mode(emergency_direction)
            # 喂数据给波形图并退出，不再执行后续普通逻辑
            self.waveform.update_chart(self.vehicle_counts)
            return
        else:
            if self.is_emergency_mode:
                # 新增检查：是否仍在锁定时间内
                if time.time() < self.emergency_mode_end_lock_time:
                    # 仍在锁定保护期内，保持当前紧急绿灯状态，不执行后续调度
                    lock_remaining = self.emergency_mode_end_lock_time - time.time()
                    self.log_label.setText(f"⚠️ 紧急模式锁定中 (保护期: 剩余 {lock_remaining:.1f}s)\n等待解除...")
                    self.waveform.update_chart(self.vehicle_counts)
                    return  # <--- 关键：锁定期间直接退出
                else:
                    # 锁定时间已过，解除紧急模式
                    self.deactivate_emergency_mode()

        # ======================================
        #  普通调度逻辑 (原有代码)
        # ======================================
        ns_pressure = self.vehicle_counts['North'] + self.vehicle_counts['South']
        ew_pressure = self.vehicle_counts['West'] + self.vehicle_counts['East']

        action, reason = TrafficAlgorithm.calculate_next_action(
            self.current_state, elapsed, ns_pressure, ew_pressure
        )

        self.log_label.setText(f"模式: 普通调度\n决策: {action}\n原因: {reason}\nNS: {ns_pressure} | EW: {ew_pressure}")
        self.waveform.update_chart(self.vehicle_counts)

        if action == 'SWITCH':
            self.switch_phase()

    #  激活紧急模式
    def activate_emergency_mode(self, direction):
        if not self.is_emergency_mode:
            self.is_emergency_mode = True
            self.flash_timer.start(500)  # 开始闪烁
            self.siren_sound.play()  # 播放音效

        # 确定目标状态
        target_state = 'ALL_RED'

        # 如果当前不是目标绿灯，强制切换 (不经过黄灯，立刻变绿)
        if self.current_state != target_state:
            self.current_state = target_state
            self.map_widget.update_lights(self.current_state)
            self.state_start_time = time.time()  # 重置计时

            # 设置锁定时间，防止灯光频繁闪烁
            self.emergency_mode_end_lock_time = time.time() + 3

            self.status_label.setText(f"🚨 紧急车辆优先: {direction} (全路口禁行) 🚨")
            self.status_label.setStyleSheet("color: red; font-weight: bold; font-size: 18px;")
            self.log_label.setText(f"⚠️ 触发紧急覆盖逻辑\n检测到救护车在 {direction}\n路口已切换至全红封锁状态！")

    #  解除紧急模式
    def deactivate_emergency_mode(self):
        self.is_emergency_mode = False
        self.flash_timer.stop()
        self.siren_sound.stop()
        self.setStyleSheet("background-color: #1e1e1e; color: white;")  # 恢复背景
        self.title_label.setStyleSheet("color: white;")

        # --- 修改核心：恢复灯光状态 ---
        # 紧急模式结束后，强制让系统进入一个确定的相位（比如南北通行），
        # 否则 current_state 停留在 'ALL_RED' 会导致普通调度逻辑判断失效。
        self.current_state = 'NS_GREEN'
        self.map_widget.update_lights(self.current_state)
        self.state_start_time = time.time()  # 重新开始计时，给新相位完整的通行时间
        # ----------------------------

        self.log_label.setText("紧急模式解除：从全红封锁恢复至 NS_GREEN 通行...")

        # 恢复正常的文字颜色
        color = "#00ff00"  # 因为上面强制设为了 NS_GREEN，所以这里直接用绿色
        self.status_label.setText(f"当前状态: {self.current_state}")
        self.status_label.setStyleSheet(f"color: {color}; font-weight: normal; font-size: 14px;")

    #  界面红黑交替闪烁效果
    def toggle_flash_effect(self):
        self.flash_state = not self.flash_state
        if self.flash_state:
            self.setStyleSheet("background-color: #330000; color: white;")  # 暗红色背景
            self.title_label.setStyleSheet("color: red;")
        else:
            self.setStyleSheet("background-color: #000000; color: white;")  # 黑色背景
            self.title_label.setStyleSheet("color: white;")

    def switch_phase(self):
        self.state_start_time = time.time()
        if self.current_state == 'NS_GREEN':
            self.current_state = 'NS_YELLOW'
        elif self.current_state == 'NS_YELLOW':
            self.current_state = 'EW_GREEN'
        elif self.current_state == 'EW_GREEN':
            self.current_state = 'EW_YELLOW'
        elif self.current_state == 'EW_YELLOW':
            self.current_state = 'NS_GREEN'
        self.map_widget.update_lights(self.current_state)
        self.status_label.setText(f"当前状态: {self.current_state}")
        color = "#00ff00" if "GREEN" in self.current_state else (
            "#ffff00" if "YELLOW" in self.current_state else "#ff0000")
        self.status_label.setStyleSheet(f"color: {color}; margin-top: 10px;")

    def closeEvent(self, event):
        for t in self.threads:
            t.stop()
        self.flash_timer.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    window = SmartTrafficCenter()
    window.show()
    sys.exit(app.exec_())