import torch
from PIL import Image
from efficientnet_pytorch import EfficientNet
import glob
import cv2
from transforms import test_transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy
checkpoint = torch.load('./model_290.pth')
print(checkpoint['image_size'])

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5)
model.load_state_dict(checkpoint['model'])
model.eval().to('cuda')
#https://github.com/jacobgil/pytorch-grad-cam/issues/95
cam = GradCAM(model,[model._conv_head] ,True)

data_list = list(glob.iglob(f'C:/Users/exper/Desktop/양산_모니터링/20231010/스페셜비엔나 260g x 2/NG_NG/*.png', recursive=True))

for image_path in data_list:
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    transform = test_transforms(512)
    input = transform(img).unsqueeze(0).to('cuda')
    preds = model(input)
    #print(preds)

    flag = True

    class_index = preds[0].argmax(dim=-1).cpu().numpy()
    # print(image_path)
    # print(int(image_path.split('\\')[-2].split('_')[0]))

    # gt_index = int(image_path.split('\\')[-2].split('_')[0])
    # flag = class_index != gt_index
    print(image_path)
    flag = class_index != 0
    if flag:
        grayscale_cam = cam(input_tensor=input)
        
        print(grayscale_cam.shape)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(numpy.asarray(img.resize((512,512))) / 255, grayscale_cam, use_rgb=False, image_weight=0.8)
        print(class_index)
        cv2.imshow("asdf", visualization)
        cv2.waitKey(0)