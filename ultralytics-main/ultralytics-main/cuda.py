import torch

print(torch.cuda.is_available())  # 检查CUDA是否可用
print(torch.cuda.current_device())  # 查看当前分配的CUDA设备
print(torch.cuda.get_device_name(0))  # 获取设备名称