import cv2
import numpy as np
from collections import defaultdict
from PIL import Image
import torch
from ultralytics import YOLO
import torchreid
from torchvision import transforms
import torch.nn.functional as F


def extract_vehicle_data_pipeline(model_path, video_path, max_frames=1):
    """
    后台处理视频流，逐帧处理（不跳帧），确保追踪稳定性
    """
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    # reid_buffer 用于存储每个 ID 对应置信度最高的抓拍
    reid_buffer = defaultdict(lambda: {'images': [], 'scores': [], 'timestamps': []})
    frame_idx = 0

    if not cap.isOpened():
        print(f"❌ 无法读取视频: {video_path}")
        return {}

    print(f"🚀 开始全量处理视频: {video_path}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_idx += 1

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.3, verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().tolist()

            for box, track_id, conf in zip(boxes, ids, confs):
                x, y, w, h = box
                # 计算坐标并防止越界
                x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)

                # 过滤掉画面边缘不完整的截图
                crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]

                if crop.size > 0 and w > 15 and h > 15:  # 适度放宽尺寸限制
                    data = reid_buffer[track_id]
                    # 策略：每个 ID 只保留全视频中置信度最高的那一帧
                    if not data['scores'] or conf > data['scores'][0]:
                        data['images'] = [crop]
                        data['scores'] = [conf]
                        data['timestamps'] = [frame_idx]

    cap.release()
    print(f"✅ 处理完成！捕获车辆总数 (独立 ID): {len(reid_buffer)}")
    return reid_buffer


# --- ReID 特征提取器 (保持单例模式) ---
class OSNetFeatureExtractor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = torchreid.models.build_model(name='osnet_x1_0', num_classes=1000, pretrained=True)
            cls._instance.model.eval()
            cls._instance.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            cls._instance.model.to(cls._instance.device)
            cls._instance.trans = transforms.Compose([
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        return cls._instance

    def extract(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        tensor = self.trans(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
            feat = F.normalize(feat, p=2, dim=1)
        return feat.cpu().numpy().flatten()


# --- 匹配逻辑 ---
def find_best_match_in_buffer(query_img, buffer, extractor):
    q_feat = extractor.extract(query_img)
    best_id, best_score, best_time, best_crop = None, -1.0, 0, None
    for tid, data in buffer.items():
        curr_feat = extractor.extract(data['images'][0])
        score = np.dot(q_feat, curr_feat)
        if score > best_score:
            best_score, best_id, best_time, best_crop = score, tid, data['timestamps'][0], data['images'][0]
    return best_id, best_score, best_time, best_crop