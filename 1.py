import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Training will run on CPU.")
x = torch.Tensor([1, 2, 3])
print(x.device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
import cv2
print(cv2.getBuildInformation()) 