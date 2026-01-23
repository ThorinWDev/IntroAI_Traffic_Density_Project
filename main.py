import os
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

# 从你提供的其他模块中导入
from video import extract_vehicle_data_pipeline, OSNetFeatureExtractor, find_best_match_in_buffer
from ui import SmartTrafficCenter
from track import TrackSystem


# ==========================================
# 1. 逻辑校验类：负责时空拓扑约束
# ==========================================
class LogicChecker:
    def __init__(self, config):
        self.config = config

    def validate(self, p_id, c_id, p_t, c_t, score):
        """
        验证跨摄像头的时间和空间逻辑
        dt: 两处抓拍的时间差
        """
        dt = c_t - p_t
        if c_id not in self.config.get(p_id, {}).get('neighbors', []):
            return 0.0, "REJECT: No Connection"

        cons = self.config[p_id]['constraints'][c_id]
        if dt < cons['t_min']:
            return 0.0, "REJECT: Too Fast"
        if dt > cons['t_max']:
            return score * 0.6, "WARN: Delayed"

        return score, "ACCEPT"


# 拓扑结构定义
TOPOLOGY = {
    '视频 1': {'neighbors': ['视频 2', '视频 3'],
             'constraints': {'视频 2': {'t_min': 0, 't_max': 9999},
                             '视频 3': {'t_min': 0, 't_max': 9999}}}
}


# ==========================================
# 2. 后台线程类
# ==========================================

class ExtractWorker(QThread):
    """专门负责处理视频 1 的抓拍池提取"""
    extracted_signal = pyqtSignal(dict)

    def __init__(self, model_path, video_path):
        super().__init__()
        self.model_path = model_path
        self.video_path = video_path

    def run(self):
        # 调用视频处理管道
        if os.path.exists(self.video_path):
            buffer = extract_vehicle_data_pipeline(self.model_path, self.video_path)
            self.extracted_signal.emit(buffer)
        else:
            print(f"未找到视频文件: {self.video_path}")
            self.extracted_signal.emit({})


class SearchWorker(QThread):
    """负责在视频 2 和 3 中进行跨镜头检索"""
    finished = pyqtSignal(list)

    def __init__(self, query_img, t_start):
        super().__init__()
        self.img = query_img
        self.t_start = t_start

    def run(self):
        res = []
        ext = OSNetFeatureExtractor()  # 使用单例 ReID 提取器
        checker = LogicChecker(TOPOLOGY)

        # 遍历下游摄像头节点
        for name, vid in {'视频 2': '2.mp4', '视频 3': '3.mp4'}.items():
            if not os.path.exists(vid):
                continue

            # 1. 提取目标视频的所有车辆数据
            buf = extract_vehicle_data_pipeline('pts/ordinary.pt', vid)

            # 2. 进行视觉特征比对
            tid, score, t_curr, crop = find_best_match_in_buffer(self.img, buf, ext)

            if tid:
                # 3. 进行时空逻辑校验
                f_score, status = checker.validate('视频 1', name, self.t_start, t_curr, score)
                if f_score > 0:
                    res.append({
                        'cam': f"{name} ({status})",
                        'time': f"Frame {t_curr}",
                        'score': f_score,
                        'crop': crop
                    })
        self.finished.emit(res)


# ==========================================
# 3. 主入口窗口
# ==========================================

class MainEntry(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智慧城市交通管理与车辆追踪系统")
        self.resize(1400, 900)

        # 页面堆栈管理器
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # 初始化两个主要子系统
        self.p1 = SmartTrafficCenter()  # 交通信号灯系统
        self.p2 = TrackSystem()  # 车辆追踪系统

        self.stack.addWidget(self.p1)
        self.stack.addWidget(self.p2)

        # 信号连接：界面切换与逻辑触发
        self.p1.switch_to_track_signal.connect(self.go_track)
        self.p2.switch_to_ui_signal.connect(lambda: self.stack.setCurrentIndex(0))
        self.p2.vehicle_selected_signal.connect(self.do_search)

    def go_track(self):
        """进入追踪模式：开始加载视频 1 并显示车辆"""
        self.stack.setCurrentIndex(1)
        self.p2.sub_stack.setCurrentIndex(0)  # 确保显示选择页

        # 启动后台线程提取视频 1 车辆数据
        self.extract_thread = ExtractWorker('pts/ordinary.pt', 'res/1.mp4')
        self.extract_thread.extracted_signal.connect(self.p2.update_vehicle_pool)
        self.extract_thread.start()

    def do_search(self, tid, img):
        """执行检索：在后续视频中寻找特定车辆"""
        # 假设从视频 1 提取时的基础时间戳为 100
        self.sw = SearchWorker(img, 100)
        self.sw.finished.connect(self.p2.show_path_results)
        self.sw.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainEntry()
    win.show()
    sys.exit(app.exec_())