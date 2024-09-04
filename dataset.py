import torch, os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):

    def __init__(self, dataset_folder, dataset_type, transform=None):
        self.dataset_folder = os.path.join(dataset_folder, dataset_type)
        self.x = [f for f in os.listdir(self.dataset_folder) if f.endswith('.jpg')]
        self.transform = transform


    def __len__(self):
        #return min(100, len(self.x))
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx, 0, 0].astype(float)
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __getitem__(self, idx):
        img_name = self.x[idx]
        img_path = os.path.join(self.dataset_folder, img_name)
        image = Image.open(img_path).convert("RGB")
        label = int(img_name.split('_')[0])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label