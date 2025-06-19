import os
import re

# 目标目录路径
target_dir = "/mnt/sda1/QHM/TANet/ckpt"

try:
    # 遍历目录中的所有文件
    for filename in os.listdir(target_dir):
        # 使用正则表达式匹配 finetune_数字.tar 格式的文件名
        match = re.match(r'finetune_(\d+)\.tar$', filename)
        if match:
            # 提取数字部分并转换为整数
            num = int(match.group(1))
            # 检查数字是否在1到279范围内
            if 1 <= num <= 279:
                # 构建完整文件路径
                file_path = os.path.join(target_dir, filename)
                try:
                    # 删除文件
                    os.remove(file_path)
                    print(f"已删除: {file_path}")
                except Exception as e:
                    print(f"删除 {file_path} 时出错: {e}")

    print("删除操作完成！")
except Exception as e:
    print(f"访问目录 {target_dir} 时出错: {e}")