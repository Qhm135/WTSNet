import os
import pandas as pd


def save_video_filenames_to_excel(folder_path, output_excel):
    # 定义常见的视频文件扩展名
    video_extensions = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.mpeg', '.mpg', '.webm']

    # 获取文件夹中的所有文件
    try:
        files = os.listdir(folder_path)
    except FileNotFoundError:
        print(f"错误：文件夹 {folder_path} 不存在")
        return
    except PermissionError:
        print(f"错误：没有权限访问文件夹 {folder_path}")
        return

    # 筛选出视频文件
    video_files = []
    for file in files:
        # 获取文件扩展名并转换为小写
        ext = os.path.splitext(file)[1].lower()
        if ext in video_extensions:
            video_files.append(file)

    # 如果没有找到视频文件
    if not video_files:
        print(f"在文件夹 {folder_path} 中没有找到视频文件")
        return

    # 创建DataFrame并保存到Excel
    df = pd.DataFrame(video_files, columns=['视频文件名'])
    try:
        df.to_excel(output_excel, index=False)
        print(f"成功将视频文件名保存到 {output_excel}")
    except Exception as e:
        print(f"保存Excel文件时出错: {e}")


# 使用示例
folder_path = r'D:\111\辅助'  # 要读取的文件夹路径
output_excel = r'D:\111\视频文件名列表.xlsx'  # 输出的Excel文件路径

save_video_filenames_to_excel(folder_path, output_excel)