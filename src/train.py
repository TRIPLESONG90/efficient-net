import glob
import torch
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler
from dataset import get_dataloader
from transforms import train_transforms, val_transforms
from efficientnet_pytorch import EfficientNet

image_size = 512
num_classes = 5
batch_size = 16
train_dir = '/dataset'
lr = 0.001
gamma = 0.9
epochs = 300
device = 'cuda'
model_type = 'efficientnet-b0'

model = EfficientNet.from_pretrained(model_type, num_classes=num_classes)

# checkpoint = torch.load('./model_70.pth')
# model.load_state_dict(checkpoint['model'])

model.to(device)


import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

import os
class_names = []
data_cnt_list = []
data_list = []
labels = []

i  =0
for class_dir in glob.iglob(f'{train_dir}/*'):
    class_name = os.path.basename(class_dir)
    class_names.append(class_name)
    files = glob.glob(f'{class_dir}/*.png')
    data_list.extend(files)
    data_cnt_list.append(len(files))
    for j in range(len(files)):
        labels.append(i)
    i += 1

total_cnt = sum(data_cnt_list)
class_weights = [total_cnt / data_cnt_list[i] for i in range(len(data_cnt_list))] 
weights = [class_weights[labels[i]] for i in range(len(data_list))]

sampler = WeightedRandomSampler(torch.DoubleTensor(weights), total_cnt)
train_loader = get_dataloader(data_list, train_transforms(image_size), batch_size=batch_size,shuffle=None, sampler=sampler)
val_loader = get_dataloader(data_list, val_transforms(image_size), batch_size=batch_size,shuffle=None, sampler=sampler)

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(val_loader)
            epoch_val_loss += val_loss / len(val_loader)
    # save model by every 10
    if (epoch + 1) % 10 == 0:
        torch.save({
            'model': model.state_dict(),
            'image_size': (image_size, image_size),
            'num_classes': num_classes,
            'transforms': val_transforms(image_size),
            'model_type' : model_type,
            'class_names' : ['OK', 'Align', 'MisAttached', 'NotAttached', 'Disheveled']
        }, f"model_{epoch + 1}.pth")

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )
torch.save({
            'model': model.state_dict(),
            'image_size': (image_size, image_size),
            'num_classes': num_classes,
            'transforms': val_transforms(image_size),
            'model_type' : model_type,
            'class_names' : ['OK', 'Align', 'MisAttached', 'NotAttached', 'Disheveled']
        }, f"model.pth")