import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap
import torch
from PIL import Image
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 初始化模型
        self.model = None

        # 设置主窗口标题和大小
        self.setWindowTitle("目标检测")
        self.setGeometry(100, 100, 800, 600)

        # 创建主布局
        layout = QVBoxLayout()

        # 创建按钮
        self.model_button = QPushButton("选择模型文件")
        self.model_button.clicked.connect(self.load_model)
        layout.addWidget(self.model_button)

        self.image_button = QPushButton("选择图片文件")
        self.image_button.clicked.connect(self.load_image)
        layout.addWidget(self.image_button)

        self.detect_button = QPushButton("检测")
        self.detect_button.clicked.connect(self.detect)
        layout.addWidget(self.detect_button)

        # 创建图片显示区域
        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        # 创建主窗口主部件
        main_widget = QWidget()
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

    def load_model(self):
        # 使用文件对话框选择模型文件
        model_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "PyTorch 模型文件 (*.pt)")
        if model_path:
            self.model = YOLO(model_path)

    def load_image(self):
        # 使用文件对话框选择图片文件
        image_path, _ = QFileDialog.getOpenFileName(self, "选择图片文件", "", "图片文件 (*.jpg *.png)")
        if image_path:
            self.image = Image.open(image_path)
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap)

    def detect(self):
        if self.model and self.image:
            results = self.model(self.image)
            img_np = np.array(self.image)
            fig, ax = plt.subplots()
            ax.imshow(img_np)

            # 分析结果
            belt_coords = []
            person_coords = []

            for r in results:
                boxes = r.boxes.xyxy.cpu()  # 将边界框信息移至CPU
                classes = r.boxes.cls.cpu()  # 将类别信息移至CPU
                class_names = r.names  # 类别名称

                # 提取belt和person的坐标信息
                for i, cls_idx in enumerate(classes):
                    class_name = class_names[int(cls_idx)]
                    box = boxes[i].numpy()
                    if class_name == 'belt':
                        belt_coords.append(box)  # 保存belt的边界框
                        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='yellow', facecolor='none')
                        ax.add_patch(rect)
                        ax.text(box[0], box[1] - 10, f'belt', color='yellow', fontsize=12, weight='bold')
                    elif class_name == 'person':
                        person_coords.append((box, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)))  # 保存person的边界框和中心坐标
                        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='blue', facecolor='none')
                        ax.add_patch(rect)
                        ax.text(box[0], box[1] - 10, f'person', color='blue', fontsize=12, weight='bold')

            # 检查行人是否越界
            for box, person in person_coords:
                person_x, person_y = person
                crossed = False
                for belt in belt_coords:
                    belt_y1, belt_y2 = belt[1], belt[3]
                    if person_y < belt_y1:  # 如果行人的中心y坐标在belt的上方
                        crossed = True
                        break

                if crossed:
                    print(f"此人{person} 已经越界！")
                    ax.text(box[0], box[1] - 30, 'Cross the border', color='red', fontsize=25, weight='bold')
                else:
                    print(f"此人{person} 在界内。")

            plt.show()
        else:
            print("请先选择模型文件和图片文件！")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
