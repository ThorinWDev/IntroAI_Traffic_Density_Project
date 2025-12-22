import os


def update_txt_files(folder_path):
    """
    遍历指定文件夹中的所有TXT文件。
    如果文件内容不为空，并且第一个字符不是'0'，则将其改为'1'。

    Args:
        folder_path (str): 包含TXT文件的文件夹路径。
    """
    print(f"开始扫描文件夹: {folder_path}\n")

    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"错误：文件夹路径不存在或不是一个目录: {folder_path}")
        return

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 构造完整的文件路径
        file_path = os.path.join(folder_path, filename)

        # 仅处理TXT文件且排除目录
        if filename.endswith(".txt") and os.path.isfile(file_path):
            print(f"正在处理文件: {filename}")

            try:
                # 1. 读取文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 检查内容是否为空
                if not content:
                    print(f"  文件为空，跳过。\n")
                    continue

                first_char = content[0]

                # 2. 检查第一个字符
                if first_char != '0':
                    # 3. 替换第一个字符为 '1'
                    new_content = '1' + content[1:]

                    print(f"  第一个字符 '{first_char}' 不是 '0'。")
                    print(f"  正在替换为 '1'。")

                    # 4. 写回文件
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)

                    print(f"  文件更新完成。\n")

                else:
                    print(f"  第一个字符是 '0'，无需修改。\n")

            except Exception as e:
                print(f"  处理文件 {filename} 时发生错误: {e}\n")

    print("--- 所有文件处理完毕 ---")


# --- 使用示例 ---
# 请将下面的 'your_folder_path' 替换为你的TXT文件所在的实际文件夹路径
your_folder_path = 'D:\\MY_FILES\\dataset\\dataset\\train\\labels'  # 替换成你的路径，例如 'C:/Users/YourName/Desktop/Labels'

# 为了演示，你可以创建一个名为 'test_data' 的文件夹，并在其中放一些txt文件
# 例如，如果你有一个名为 'example.txt' 的文件，内容如你所提供：
# 4 0.4699519230769231 0.47475961538461536 0.8545673076923077 0.9495192307692307
# 运行后，其内容会变为：
# 1 0.4699519230769231 0.47475961538461536 0.8545673076923077 0.9495192307692307

# 请确保在你运行代码之前设置正确的文件夹路径!
update_txt_files(your_folder_path)