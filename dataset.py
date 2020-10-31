import torch
from torch.utils.data import DataLoader, Dataset
import glob
import os
from PIL import Image
from torchvision import models, transforms


class MyDataSet(Dataset):
    def __init__(self, root_dir='./data', train_val='train', transform=None):
        self.data_path = os.path.join(root_dir, train_val)
        self.image_names = glob.glob(self.data_path + '/*/*.jpg')
        self.image_names = [name for name in self.image_names if name.split('/')[-1] != '.DS_Store']
        self.data_transform = transform
        self.train_val = train_val
        self.name_dict = {"apple": 0, "banana": 1, "grape": 2, "orange": 3, "pear": 4}

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, item):
        img_path = self.image_names[item]
        img = Image.open(img_path)
        image = img
        label = img_path.split('/')[-2]
        label = int(self.name_dict[label])
        if self.data_transform is not None:
            try:
                image = self.data_path
                image = self.data_transform[self.train_val](img)
            except Exception as e:
                print('can not load image:{}'.format(img_path))
                print(e)
        return image, label


if __name__ == '__main__':
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {x: MyDataSet(
        root_dir='./data',
        train_val=x,
        transform=data_transforms
    ) for x in ['train', 'val']}

    dataloaders = dict()
    dataloaders['train'] = DataLoader(
        image_datasets['train'],
        batch_size=32,
        shuffle=True
    )
    dataloaders['val'] = DataLoader(
        image_datasets['val'],
        batch_size=32,
        shuffle=True
    )

    data1 = iter(dataloaders['train'])

    for i in range(1):
        img, label = next(data1)
        print('image: ', img)
        print('label: ', label)
