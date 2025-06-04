import torch

# 定义你的模型架构
# model = ...

model = 
# 加载模型的checkpoint
checkpoint_path = '你的checkpoint路径.pth'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# 从checkpoint中恢复模型、优化器和其他状态
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
epoch = checkpoint['epoch']
loss_scaler.load_state_dict(checkpoint['scaler'])
args = checkpoint['args']
ExactMatchRatio = checkpoint['ExactMatchRatio']

# 打印保存的args参数
print(args)
