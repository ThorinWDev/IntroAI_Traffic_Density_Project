import os
import glob
import logging

# --- 配置参数 ---
# 图片文件所在的根目录
IMAGE_DIR = 'D:\\MY_FILES\\my_ambulance_or_others_DATASET\\valid\\images'
# 标签文件所在的根目录
LABEL_DIR = 'D:\\MY_FILES\\my_ambulance_or_others_DATASET\\valid\\labels'
# 图片文件的扩展名 (例如：.jpg, .png, .jpeg)
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp']
# 标签文件的扩展名 (例如：.txt, .xml, .json)
LABEL_EXTENSION = '.txt'

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def synchronize_files(image_dir, label_dir, image_exts, label_ext):
    """
    检查标签文件是否有对应的图片文件，如果没有则删除标签文件。
    """
    if not os.path.isdir(image_dir):
        logging.error(f"图片目录不存在: {image_dir}")
        return
    if not os.path.isdir(label_dir):
        logging.error(f"标签目录不存在: {label_dir}")
        return

    logging.info(f"开始同步数据集：\n图片目录: {image_dir}\n标签目录: {label_dir}")

    # 1. 获取所有标签文件的路径
    # 使用 os.path.join 确保路径兼容性
    search_pattern = os.path.join(label_dir, f'*{label_ext}')
    label_files = glob.glob(search_pattern)

    total_labels = len(label_files)
    deleted_count = 0

    logging.info(f"找到 {total_labels} 个 {label_ext} 标签文件。")

    # 2. 遍历每个标签文件
    for label_path in label_files:
        # 获取文件名（不带扩展名）
        base_name = os.path.splitext(os.path.basename(label_path))[0]

        # 3. 检查对应的图片文件是否存在
        image_found = False

        for ext in image_exts:
            image_path = os.path.join(image_dir, base_name + ext)
            if os.path.exists(image_path):
                image_found = True
                break  # 找到图片后跳出扩展名循环

        # 4. 如果没有找到对应的图片，则删除标签文件
        if not image_found:
            try:
                os.remove(label_path)
                deleted_count += 1
                logging.warning(f"已删除无对应图片的标签文件: {label_path}")
            except Exception as e:
                logging.error(f"删除标签文件失败 {label_path}: {e}")

    logging.info("--- 同步完成 ---")
    logging.info(f"总标签文件数（初始）：{total_labels}")
    logging.info(f"已删除的标签文件数：{deleted_count}")
    logging.info(f"剩余的标签文件数：{total_labels - deleted_count}")


# 运行函数
# if __name__ == "__main__":
    # --- 重要的：请修改这里的路径！ ---
    # 示例：
    # IMAGE_DIR = '/home/user/data/project/images'
    # LABEL_DIR = '/home/user/data/project/labels'

    # 运行同步
synchronize_files(IMAGE_DIR, LABEL_DIR, IMAGE_EXTENSIONS, LABEL_EXTENSION)