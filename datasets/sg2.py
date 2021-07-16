import os
import numpy as np
import torchvision.datasets as datasets
from sklearn.preprocessing import MinMaxScaler


class StyleGAN2_Data(datasets.ImageFolder):

    def __init__(self, root='data', split='train', latent_dim=512):
        super(StyleGAN2_Data, self).__init__(root)

        assert os.path.exists(root), "root: {} not found.".format(root)

        self.root = os.path.join(root, split)
        self.split = split
        self.scaler = MinMaxScaler(feature_range = (0, 1))
        self.latent_dim = latent_dim

        self.files = np.load(os.path.join(root, split, 'npy', 'files.npy'))

        # self.labels = np.load(os.path.join(root, split, 'npy', 'landmarks.npy'))
        if split == 'train' or split == 'train_all':
            self.labels_original = np.load(os.path.join(root, split, 'npy', 'landmarks.npy'))
            self.labels = self.scale_label(self.labels_original)

        elif split == 'val' or 'test':
            self.labels_original = np.load(os.path.join(root, 'train_all', 'npy', 'landmarks.npy'))
            self.scale_label(self.labels_original)

            self.labels_original = np.load(os.path.join(root, split, 'npy', 'landmarks.npy'))
            self.labels = self.scale_val_label(self.labels_original)
        else:
            raise ValueError(f"split was not set correctly split = ['train', 'val', 'test'] not {split}")

        assert len(self.files) == len(self.labels), f"[INFO] len(files)={len(self.files)} != len(labels)={len(self.files)}"


    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        seed = int(self.files[index].replace('images/seed', '').replace('.png', ''))
        latent = np.random.RandomState(seed).randn(self.latent_dim).astype(np.float32)
        label = self.labels[index]

        out = {
            'latent': latent,
            'label': label,
            'seed': seed,
            'index': index
            }

        return out


    def scale_label(self, labels):
        return self.scaler.fit_transform(labels)


    def scale_val_label(self, labels):
        return self.scaler.transform(labels)


    def inv_scale_label(self, labels):
        return np.rint(self.scaler.inverse_transform(labels))
        