from ultralytics import YOLO

def train_model():
    model = YOLO(r"D:\project\ultralytics-main\ultralytics-main\yolov8n.pt")
    model.train(
        model=r"D:\project\ultralytics-main\ultralytics-main\yolov8n.pt",
        data=r"D:\project\ultralytics-main\ultralytics-main\ultralytics\cfg\datasets\mydate.yaml",
        epochs=9,
        imgsz=640,
        batch=2
    )

if __name__ == '__main__':
    train_model()
