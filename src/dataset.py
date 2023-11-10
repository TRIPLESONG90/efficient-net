from PIL import Image
from torch.utils.data import DataLoader, Dataset

from transforms import train_transforms

class StrapDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength
        # df
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        img_transformed = self.transform(img)

        label = img_path.split('/')[-2]
        label = int(label.split('_')[0])

        return img_transformed, label
    
def get_dataloader(data_list, transform, batch_size=32, shuffle=True, sampler = None):
    #asdf
    data = StrapDataset(data_list, transform=transform)
    loader = DataLoader(dataset = data, batch_size=batch_size, shuffle=shuffle, sampler=sampler)
    return loader
