import torch
from efficientnet_pytorch import EfficientNet
import numpy as np
from PIL import Image
from torchvision import transforms as t

checkpoint = torch.load('./model_0.pth')

model = EfficientNet.from_pretrained(checkpoint['model_type'], num_classes=checkpoint['num_classes'])
model.load_state_dict(checkpoint['model'])
model.eval().to('cuda')
#ㅁㄴㅇㅍㅁㄴㅇㅁㄴㅇㅍㄴㅁㅇㅍㅍ
while True:
    tmp = np.zeros((checkpoint['image_size'][0],checkpoint['image_size'][1], 3), np.uint8)
    tmp = Image.fromarray(tmp)
    tmp = checkpoint['transforms'](tmp).unsqueeze(0).to('cuda')
    preds = model(tmp)
    print(preds)