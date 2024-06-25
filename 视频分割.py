import cv2
import os

def video_to_frames(video_path, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 保存当前帧为图像
        frame_filename = os.path.join(output_folder, f'frame_{frame_count:05d}.jpg')
        cv2.imwrite(frame_filename, frame)
        print(f"保存帧 {frame_count} 为文件 {frame_filename}")
        frame_count += 1

    cap.release()
    print(f'视频处理完成，共提取了 {frame_count} 帧图像。')

# 使用示例
video_path = r"D:\资料\人员越界视频\人员越界视频\跨越皮带.mp4"  # 替换为你的视频文件路径
output_folder = r"D:\project\ultralytics-main\ultralytics-main\井下图片"  # 替换为你想保存图像的文件夹路径
video_to_frames(video_path, output_folder)
