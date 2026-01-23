import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

def extract_vehicle_data_pipeline(model_path, video_path, max_frames=1):
    """
    功能：后台处理视频流，提取每辆车置信度最高的关键帧。
    优化点：下调了置信度阈值，放宽了尺寸限制。
    """

    # --- 1. 初始化 ---
    print(f"🔄 正在加载模型: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 视频打不开: {video_path}")
        return {}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 使用字典存储追踪到的车辆数据
    reid_buffer = defaultdict(lambda: {'images': [], 'scores': [], 'timestamps': []})
    frame_idx = 0
    print("🚀 正在后台处理视频，请稍候...")

    # --- 2. 主循环 (后台处理) ---
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_idx += 1

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.7,
            verbose=False
        )

        # 检查是否有检测到目标
        if results[0].boxes.id is not None:
            boxes_xywh = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()

            for box, track_id, conf in zip(boxes_xywh, track_ids, confidences):
                x_c, y_c, w_box, h_box = box
                x1 = int(max(0, x_c - w_box / 2))
                y1 = int(max(0, y_c - h_box / 2))
                x2 = int(min(width, x_c + w_box / 2))
                y2 = int(min(height, y_c + h_box / 2))

                vehicle_crop = frame[y1:y2, x1:x2]

                # 核心优化 2：放宽尺寸限制。从 20 像素降到 10 像素
                if vehicle_crop.size > 0 and w_box > 10 and h_box > 10:
                    data = reid_buffer[track_id]

                    if len(data['images']) < max_frames:
                        data['images'].append(vehicle_crop)
                        data['scores'].append(conf)
                        data['timestamps'].append(frame_idx)
                    else:
                        # 仅保留最高置信度的帧
                        min_idx = np.argmin(data['scores'])
                        if conf > data['scores'][min_idx]:
                            data['images'][min_idx] = vehicle_crop
                            data['scores'][min_idx] = conf
                            data['timestamps'][min_idx] = frame_idx

    # --- 3. 释放资源 ---
    cap.release()
    cv2.destroyAllWindows()

    print(f"✅ 处理完成！总计捕获独立 ID 数量: {len(reid_buffer)}")
    return reid_buffer