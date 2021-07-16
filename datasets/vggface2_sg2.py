import os
import numpy as np
from PIL import Image
import torchvision.datasets as datasets
from sklearn.preprocessing import MinMaxScaler

INPUT_SIZE = 224

class StyleGAN2_Data(datasets.ImageFolder):

    def __init__(self, root='data/', split='train', transform=None, scale_size=-1):
        super(StyleGAN2_Data, self).__init__(root)

        assert os.path.exists(root), "root: {} not found.".format(root)

        self.root = os.path.join(root, split)
        self.split = split
        self.transform = transform
        self.scaler = MinMaxScaler(feature_range = (-1, 1))
        self.scale_size = scale_size
    
        if split == 'train':
            self.labels_original = np.load(os.path.join(root, 'train', 'npy', 'landmarks.npy'))
            if scale_size > 0:
                self.labels = self.scale_label(self.labels_original / scale_size * INPUT_SIZE)
            else:
                self.labels = self.scale_label(self.labels_original)

        elif split == 'val' or 'test':
            self.labels_original = np.load(os.path.join(root, 'train', 'npy', 'landmarks.npy'))
            if scale_size > 0:
                self.scale_label(self.labels_original / scale_size * INPUT_SIZE)
            else:
                self.scale_label(self.labels_original)

            self.labels_original = np.load(os.path.join(root, split, 'npy', 'landmarks.npy'))
            if scale_size > 0:
                self.labels = self.scale_val_label(self.labels_original / scale_size * INPUT_SIZE)
            else:
                self.labels = self.scale_val_label(self.labels_original)
        else:
            raise ValueError(f"split was not set correctly split = ['train', 'val', 'test'] not {split}")

        self.files = np.load(os.path.join(root, split, 'npy', 'files.npy'))

        assert len(self.files) == len(self.labels), f"[INFO] len(files)={len(self.files)} != len(labels)={len(self.files)}"


    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        path = os.path.join(self.root, self.files[index])
        label = self.labels[index]
        img = Image.open(path).convert("RGB")
        img_size_ori = img.size

        img = self.transform(img)
        img_size = (img.shape[1], img.shape[2])

        out = {
            'image': img,
            'label': label,
            'meta': {
                        'im_size_ori': img_size_ori,
                        'im_size': img_size,
                        'index': index
                    }
            }

        return out


    def get_image(self, index, root):
        path = os.path.join(root, self.files[index])
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        return img


    def scale_label(self, labels):
        return self.scaler.fit_transform(labels)


    def scale_val_label(self, labels):
        return self.scaler.transform(labels)


    def inv_scale_label(self, labels):
        if self.scale_size > 0:
            return np.rint(self.scaler.inverse_transform(labels) * self.scale_size / INPUT_SIZE)
        else:
            return np.rint(self.scaler.inverse_transform(labels))
        