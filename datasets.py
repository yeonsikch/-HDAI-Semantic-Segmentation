from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import os
import cv2
import glob
import numpy as np

class HeartDataset(Dataset):
    def __init__(self, path, mode = "train", domain='all', transforms = None):
        self.mode = mode
        self.path = path
        if mode != 'test':
            self.path = os.path.join(path, mode)
        self.domain = domain.upper()

        self.transforms = transforms
        self.img_list = []
        self.mask_list = []
        self._get_img_mask_paths()

    def _get_img_mask_paths(self):
        if self.domain=='ALL':
            for t in (["A2C", "A4C"]):
                self.img_list.extend(glob.glob(os.path.join(self.path, t, '*.png')))
                self.mask_list.extend(glob.glob(os.path.join(self.path, t, '*.npy')))
        else:
            self.img_list.extend(glob.glob(os.path.join(self.path, self.domain, '*.png')))
            self.mask_list.extend(glob.glob(os.path.join(self.path, self.domain, '*.npy')))

        self.img_path_list = [im for im in self.img_list]
        self.mask_path_list = [msk for msk in self.mask_list]
        self.img_path_list.sort()
        self.mask_path_list.sort()

    def __getitem__(self, index):
        image = cv2.imread(str(self.img_path_list[index]), cv2.IMREAD_UNCHANGED)
        mask = np.load(self.mask_path_list[index])
        height, width, _ = image.shape
        augmented = self.transforms(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask'].unsqueeze(0)

        if self.mode == 'test':
            return image, mask, height, width
        else:
            return image, mask

    def __len__(self):
        return len(self.img_path_list)

        
def train_aug():
    return A.Compose([
        A.Resize(height=416, width=416, p=1.0),
        #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.RandomRotate90(),
        ToTensorV2(p=1.0),
    ], p=1.0)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    data = HeartDataset("../dataset",transforms = train_aug())
    image, mask = data[0]
    train_loader = DataLoader(data, batch_size=8 , shuffle=True, num_workers=4, pin_memory=True)
    print(len(train_loader))
    print(image.shape, mask.shape)
    cv2.imwrite('./data.png', np.array(image.permute(1,2,0)))
    cv2.imwrite('./mask.png', np.array(mask.permute(1,2,0)))
