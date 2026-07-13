import torch

# 加载 checkpoint
checkpoint = torch.load("Fusion_Denoise_Network/experiments/mamba8-1-10-16-1.5-batch4/weights/checkpoint_Diff_TDN.pth", map_location="cpu")

# 查看 keys（参数的层名）
print(checkpoint.keys())  # 可能包含 'state_dict' 或直接是权重
# 获取模型的 state_dict
model_state_dict = checkpoint['model']

# 打印所有层的名称
for key in model_state_dict.keys():
    print(key)
