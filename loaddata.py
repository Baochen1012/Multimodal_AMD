import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision import transforms

script_dir = os.path.dirname(os.path.abspath(__file__))

class CrossPairedDataset(Dataset):
    def __init__(self, oct_root, cfp_root, mode='cross', transform=None, device=None, resample=True):
        self.oct_root = oct_root
        self.cfp_root = cfp_root
        self.mode = mode  # 'cross' for cross pairing, 'pair' for original pairing
        print(f"Paths: {oct_root.replace(script_dir, '')}, {cfp_root.replace(script_dir, '')} ")
        self.transform = transform
        self.device = device
        print("Finding classes...")
        self.classes, self.class_to_idx = self._find_classes(oct_root)
        print(f"Found {len(self.classes)} classes.")
        print("Making dataset...")
        self.samples = self._make_dataset()
        self.resample = resample

        if self.resample:
            print("Resampling classes...")
            self.samples = self._resample_classes()
            print(f"Total samples after resampling: {len(self.samples)}")

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        print(f"Class_to_idx:{class_to_idx}")
        return classes, class_to_idx

    def _make_dataset(self):
        samples = []
        for target_class in sorted(self.class_to_idx.keys()):
            print(f"Processing class: {target_class}")
            target_dir_oct = os.path.join(self.oct_root, target_class)
            target_dir_cfp = os.path.join(self.cfp_root, target_class)
            if not os.path.isdir(target_dir_oct) or not os.path.isdir(target_dir_cfp):
                print(f"Directory missing for class: {target_class}, skipping...")
                continue

            # Mapping image names in CFP to OCT based on the unique identifier
            cfp_images = {self._extract_unique_id(name): name for name in os.listdir(target_dir_cfp)}

            for root, _, fnames in sorted(os.walk(target_dir_oct, followlinks=True)):
                for fname in sorted(fnames):
                    unique_id = self._extract_unique_id(fname)
                    # print('unique_id',unique_id)
                    if self.mode == 'cross':
                        # Cross pairing case
                        for cfp_name in cfp_images.values():
                            path_oct = os.path.join(root, fname)
                            path_cfp = os.path.join(target_dir_cfp, cfp_name)
                            # print(path_oct,path_cfp)
                            item = (path_oct, path_cfp, self.class_to_idx[target_class])
                            samples.append(item)
                    elif self.mode == 'pair' and unique_id in cfp_images:
                        # Original pairing case
                        path_oct = os.path.join(root, fname)
                        path_cfp = os.path.join(target_dir_cfp, cfp_images[unique_id])
                        # print(path_oct,path_cfp)
                        item = (path_oct, path_cfp, self.class_to_idx[target_class])
                        samples.append(item)

        if self.mode == 'cross': #新加的，为了制造随机数减少训练样本数量防止过拟合，从219195到10000
            random.seed(0)  # 初始化随机数生成器的种子，以确保结果的可重复性。因为是伪随机：指定随机数生成器开始的起点，固定算法挑选的样例。
            n = 2000
            samples = random.sample(samples, n)
        print(f"Total samples collected: {len(samples)}")
        return samples
    
    def _extract_unique_id(self, filename):
        # Strip extension, remove first two characters, and split by underscore
        return filename[:-4][2:].split('_')[0]

    def __len__(self):
        return len(self.samples)

    def _get_class_sample_counts(self):
        # Get the count of samples for each class
        class_counts = {cls: 0 for cls in self.class_to_idx}
        for _, _, cls_idx in self.samples:
            class_name = self.classes[cls_idx]
            class_counts[class_name] += 1
        return class_counts

    def _resample_classes(self):
        # Perform resampling to handle class imbalance  执行重采样以处理类别不平衡
        class_counts = self._get_class_sample_counts()
        max_count = max(class_counts.values())

        new_samples = []
        for class_name, count in class_counts.items():
            print(class_name,count)
            class_samples = [s for s in self.samples if s[2] == self.class_to_idx[class_name]]
            if count < max_count and count > 0:
                # Upsample: Randomly duplicate samples
                resampled_samples = class_samples * (max_count // count) + random.choices(class_samples,
                                                                                          k=max_count % count)
            else:
                # No need to downsample as we want at least max_count samples
                resampled_samples = class_samples
            new_samples.extend(resampled_samples)

        return new_samples

    @staticmethod
    def Resize(img):
        import numpy as np
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        img = img.resize((224, 224))
        img = np.array(img) / 255.
        assert img.shape == (224, 224, 3)
        img = img - imagenet_mean
        img = img / imagenet_std
        return img

    def __getitem__(self, idx):
        # print(f"Fetching data for index: {idx}")
        path_oct, path_cfp, target = self.samples[idx]
        oct_image = Image.open(path_oct).convert("RGB")
        oct_image = self.__class__.Resize(oct_image)
        cfp_image = Image.open(path_cfp).convert("RGB")
        cfp_image = self.__class__.Resize(cfp_image)

        # 应用transform（包括调整大小、转换为Tensor、归一化）
        if self.transform:
            oct_image = self.transform(oct_image)
            cfp_image = self.transform(cfp_image)

        oct_image, cfp_image = oct_image.to(self.device), cfp_image.to(self.device)
        target = torch.tensor(target, dtype=torch.long, device=self.device)  # 使用long类型tensor

        return oct_image, cfp_image, target


def get_train_val_test_loaders(dataset, train_ratio=0.7, val_ratio=0.15, batch_size=32, shuffle=True, num_workers=4):
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    print(f"Training size: {train_size}")
    print(f"Validation size: {val_size}")
    print(f"Test size: {test_size}")

    print(f"Number of batches in training loader: {len(train_dataset) // batch_size}")
    print(f"Number of batches in validation loader: {len(val_dataset) // batch_size}")
    print(f"Number of batches in test loader: {len(test_dataset) // batch_size}")

    return train_loader, val_loader, test_loader

if __name__=='__main__':

    print('数据集情况')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()

    # oct_root = os.path.join(script_dir, 'oct_data/train')
    # cfp_root =  os.path.join(script_dir, 'cfp_data/train')
    # traindataset = CrossPairedDataset(oct_root, cfp_root, mode='cross',transform=transform, device=device)
    # trainloader = DataLoader(traindataset, batch_size=32, shuffle=True)
    #
    # oct_root = os.path.join(script_dir, 'oct_data/test')
    # cfp_root =  os.path.join(script_dir, 'cfp_data/test')
    # testdataset = CrossPairedDataset(oct_root, cfp_root, mode = 'pair',transform=transform, device=device)
    # testloader = DataLoader(testdataset, batch_size=32, shuffle=True)
    #
    # oct_root = os.path.join(script_dir, 'oct_data/val')
    # cfp_root =  os.path.join(script_dir, 'cfp_data/val')
    # valdataset = CrossPairedDataset(oct_root, cfp_root, mode='pair',transform=transform, device=device)
    # valloader = DataLoader(testdataset, batch_size=32, shuffle=True)

    oct_root = os.path.join(script_dir, 'oct_data/all')
    cfp_root =  os.path.join(script_dir, 'cfp_data/all')
    alldataset = CrossPairedDataset(oct_root, cfp_root, mode='pair',transform=transform, device=device)
    train_loader, val_loader, test_loader = get_train_val_test_loaders(alldataset, train_ratio=0.7, val_ratio=0.15, batch_size=32, shuffle=True, num_workers=4)
    #shuffle=True  在每个epoch内图像会被打乱

    #cross 交叉配对，pair按照Id一一配对

    # import pandas as pd
    # tr = pd.read_excel('pair_view.xlsx')
    # tr['dataset'] = 'train'
    # te = pd.read_excel('test_view.xlsx')
    # te['dataset'] = 'test'
    # va = pd.read_excel('val_view.xlsx')
    # va['dataset'] = 'val'
    # al = pd.read_excel('all_view.xlsx')
    # al = al.applymap(lambda x:str(x).replace('d:\Gusoku\Code\troch\retfound','.'))
    # tri = pd.concat([tr,te,va],axis = 0).applymap(lambda x:str(x).replace('D:\Gusoku\Code\troch\retfound','.'))
    # print(tri)
    # print(al)
    # tri.to_excel('tri_view.xlsx',index=False)
    # al.to_excel('mix_view.xlsx',index=False)



