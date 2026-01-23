import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


class TrackSystem(QWidget):
    # 信号：当选定某辆车后，传出该车的 ID 和图像给 Main 进行全局比对
    vehicle_selected_signal = pyqtSignal(str, object)
    # 信号：返回主监控界面
    switch_to_ui_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setStyleSheet("background-color: #1e1e1e; color: white;")
        self.layout = QVBoxLayout(self)

        # --- 1. 顶部导航栏 ---
        top_bar = QHBoxLayout()
        self.btn_back = QPushButton("⬅ 返回监控中心")
        self.btn_back.setFixedSize(150, 40)
        self.btn_back.setStyleSheet("background-color: #444; border-radius: 5px;")
        self.btn_back.clicked.connect(self.handle_back_logic)
        top_bar.addWidget(self.btn_back)
        top_bar.addStretch()
        self.layout.addLayout(top_bar)

        # --- 2. 页面堆栈 ---
        self.sub_stack = QStackedWidget()
        self.layout.addWidget(self.sub_stack)

        # 页面 A: 车辆抓拍池 (主要由 1.mp4 提取)
        self.page_selection = QWidget()
        sel_layout = QVBoxLayout(self.page_selection)
        self.pool_grid = QGridLayout()
        scroll_sel = QScrollArea()
        scroll_sel.setWidgetResizable(True)
        container_sel = QWidget()
        container_sel.setLayout(self.pool_grid)
        scroll_sel.setWidget(container_sel)
        sel_layout.addWidget(QLabel("【步骤1】请从 视频 1 中选择要追踪的目标车辆："))
        sel_layout.addWidget(scroll_sel)
        self.sub_stack.addWidget(self.page_selection)

        # 页面 B: 拓扑轨迹展示 (展示在 2.mp4, 3.mp4 中的匹配结果)
        self.page_path = QWidget()
        path_layout = QVBoxLayout(self.page_path)
        self.path_v_box = QVBoxLayout()
        self.path_v_box.setAlignment(Qt.AlignTop)
        scroll_path = QScrollArea()
        scroll_path.setWidgetResizable(True)
        container_path = QWidget()
        container_path.setLayout(self.path_v_box)
        scroll_path.setWidget(container_path)
        path_layout.addWidget(QLabel("【步骤2】跨镜头追踪结果 (基于时空拓扑约束及视觉重识别)："))
        path_layout.addWidget(scroll_path)
        self.sub_stack.addWidget(self.page_path)

    # --- 核心辅助函数
    def np_to_pixmap(self, img):
        """将 OpenCV/Numpy 格式图像转换为 PyQt 可显示的 QPixmap [cite: 78]"""
        if img is None:
            return QPixmap()
        try:
            # 确保数组在内存中是连续的，解决之前的 TypeError
            img = np.ascontiguousarray(img)
            h, w, c = img.shape
            bytesPerLine = 3 * w
            # 将 numpy 数据转为字节流并创建 QImage
            q_img = QImage(img.tobytes(), w, h, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            return QPixmap.fromImage(q_img)
        except Exception as e:
            print(f"❌ 图片转换出错: {e}")
            return QPixmap()

    def handle_back_logic(self):
        """返回逻辑：如果在路径页则回退到选择页，否则返回主 UI"""
        if self.sub_stack.currentIndex() == 1:
            self.sub_stack.setCurrentIndex(0)
        else:
            self.switch_to_ui_signal.emit()

    # --- 接口 1：接收 1.mp4 提取的抓拍数据并刷新 UI ---
    def update_vehicle_pool(self, buffer):
        # 清空旧的网格内容
        for i in reversed(range(self.pool_grid.count())):
            self.pool_grid.itemAt(i).widget().setParent(None)

        if not buffer:
            self.pool_grid.addWidget(QLabel("⚠️ 视频 1 中未发现有效车辆轨迹"), 0, 0)
            return

        for i, (tid, data) in enumerate(buffer.items()):
            btn = QPushButton()
            btn.setFixedSize(160, 200)
            v_box = QVBoxLayout(btn)

            # 显示抓拍到的车辆截图 [cite: 56, 76]
            img_lbl = QLabel()
            pix = self.np_to_pixmap(data['images'][0])  # 取置信度最高帧
            img_lbl.setPixmap(pix.scaled(140, 110, Qt.KeepAspectRatio))
            img_lbl.setAlignment(Qt.AlignCenter)

            info_lbl = QLabel(f"轨迹 ID: {tid}\n置信度: {data['scores'][0]:.2f}")
            info_lbl.setAlignment(Qt.AlignCenter)

            v_box.addWidget(img_lbl)
            v_box.addWidget(info_lbl)

            # 绑定点击事件，进入跨视频检索流程
            btn.clicked.connect(lambda ch, t=tid, im=data['images'][0]: self.select_vehicle(t, im))
            self.pool_grid.addWidget(btn, i // 4, i % 4)

    def select_vehicle(self, tid, img):
        """选中车辆后切换页面并发出检索信号"""
        self.vehicle_selected_signal.emit(str(tid), img)
        self.sub_stack.setCurrentIndex(1)
        # 初始化路径页显示
        for i in reversed(range(self.path_v_box.count())):
            self.path_v_box.itemAt(i).widget().setParent(None)
        self.path_v_box.addWidget(QLabel(f"🔍 正在从视频 2 和 视频 3 中检索目标车 (ID: {tid})..."))

    # --- 接口 2：展示来自 2.mp4, 3.mp4 的拓扑追踪路径及截图 ---
    def show_path_results(self, results):
        """
        展示拓扑图节点。根据《逻辑检查》报告，此处需体现物理约束校验结果 。
        """
        # 清空等待提示
        for i in reversed(range(self.path_v_box.count())):
            self.path_v_box.itemAt(i).widget().setParent(None)

        if not results:
            self.path_v_box.addWidget(QLabel("❌ 检索结束：未发现符合时空逻辑的匹配轨迹。"))
            return

        # 1. 绘制起始节点
        start_node = QLabel("🏁 起点节点：视频 1 (原始抓拍)")
        start_node.setStyleSheet("color: #00FFCC; font-weight: bold;")
        self.path_v_box.addWidget(start_node)

        # 2. 依次绘制匹配到的视频节点
        for res in results:
            # 绘制拓扑连接箭头
            arrow = QLabel("⬇")
            arrow.setAlignment(Qt.AlignCenter)
            arrow.setStyleSheet("font-size: 20px; color: #666;")
            self.path_v_box.addWidget(arrow)

            # 绘制结果卡片
            card = QFrame()
            card.setStyleSheet("background: #2b2b2b; border: 2px solid #0078d7; border-radius: 10px; padding: 10px;")
            row = QHBoxLayout(card)

            # 左侧：文字信息（体现逻辑修正后的得分 [cite: 87]）
            info = QVBoxLayout()
            info.addWidget(QLabel(f"🎥 位置：{res['cam']}"))
            info.addWidget(QLabel(f"🕒 时间点：{res['time']}"))
            info.addWidget(QLabel(f"📈 最终评分：{res['score']:.2f}"))
            row.addLayout(info)

            # 右侧：展示从该视频中扣出来的匹配车辆截图 [cite: 76]
            if res.get('crop') is not None:
                crop_lbl = QLabel()
                pix = self.np_to_pixmap(res['crop'])
                crop_lbl.setPixmap(pix.scaled(120, 100, Qt.KeepAspectRatio))
                crop_lbl.setStyleSheet("border: 1px solid white;")
                row.addWidget(crop_lbl)

            self.path_v_box.addWidget(card)