import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torchvision.transforms as tr
import torch.nn.functional as F
import torch.nn as nn
import latex_generator as gen
import os
import pickle

def create_data_iterator(root, is_train=True, batch_size=16):
    dataset = LatexImgDataset(root, is_train)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )


class LatexImgDataset(Dataset):

    def __init__(self, root, train=True):
        super(LatexImgDataset, self).__init__()

        self.root = root
        self.gen = gen.ExpressionGenerator(25)
        self.eos_index = self.gen.token2id['</s>']
        self.sos_index = self.gen.token2id['<s>']
        self.pad_index = self.gen.token2id['<pad>']
        self.train = train
        self.transform = tr.ToTensor()
        data_path = os.path.join(self.root, 'train_data' if train else 'test_data')
        with open(data_path, "rb") as f:
            content = pickle.load(f)
            self.data = content['images']
            self.targets = content.get('targets')

    def __getitem__(self, index):
        img = 255 - self.data[index]
        img = self.transform(img)
        target = self.targets[index]
        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def collate_fn(self, elements):
        x, y = zip(*elements)
        y = [torch.LongTensor([self.gen.token2id[w] for w in seq if w in self.gen.token2id]) for seq in y]
        x = self.image_to_batch(x)
        y, y_len = self.target_to_batch(y)
        return x, y

    def target_to_batch(self, sequences):
        lengths = torch.LongTensor([len(s) + 2 for s in sequences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.pad_index)
        sent[0] = self.sos_index
        for i, s in enumerate(sequences):
            sent[1:lengths[i] - 1, i].copy_(s)
            sent[lengths[i] - 1, i] = self.eos_index
        return sent.T, lengths

    def image_to_batch(self, images):
        sizes = torch.LongTensor([img.shape for img in images])

        max_size = sizes.max(dim=0).values
        max_h = max_size[1].item()
        max_w = max_size[2].item()

        new_images = []

        for i, img in enumerate(images):
            h, w = sizes[i][1].item(), sizes[i][2].item()
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
            top_pad = (max_h - h) //2
            bottom_pad = max_h - h - top_pad
            pd = (left_pad, right_pad, top_pad, bottom_pad)
            new_images.append(F.pad(img, pd, "constant", 0.0).unsqueeze(0))
        return torch.cat(new_images)
