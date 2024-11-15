#!/usr/bin/env python
# coding: utf-8
import torch
import numpy as np
import sys
import scipy.stats
import time

class LinearEncoder:
    def __init__(self, dim=10000, num=256):
        self.dim = dim
        self.num = num

    def get_hdv(self, dim=10000, num=1):
        assert num > 0, "[Error] Only support nonzero size in get_hdv()"
        if num == 1:
            return np.random.choice([-1, 1], size=dim)
        else:
            result = np.random.randint(2, size=(num, dim))
            result = (result - 0.5) * 2
            return result.astype('int')

    def build_item_mem(self):
        assert self.num > 1, "No need of this function if only one vector in the item memory."
        print("generating linear item memory...")
        item_mem = self.get_hdv(dim=self.dim, num=self.num)
        index = np.arange(self.dim // 2)
        np.random.shuffle(index)
        interval = int((self.dim / 2) / (self.num - 1))
        pointer = 0
        for i in range(1, self.num):
            new_item = np.copy(item_mem[i - 1])
            if i == self.num - 1:
                new_item[index[pointer:]] *= -1
            else:
                new_item[index[pointer: pointer + interval]] *= -1
            pointer += interval
            item_mem[i] = new_item
        self.item_mem = torch.from_numpy(item_mem)
        return self.item_mem

    def encode_one_img(self, x):
        rv = self.item_mem[x[0]]
        for i in range(1, x.shape[0]):
            rv = torch.roll(rv, i)
            rv = -rv * self.item_mem[x[i]]
        return rv

    def encode_data_extract_labels(self, datast):
        n = len(datast)
        rv = torch.zeros((n, self.dim))
        labels = torch.zeros(n).long()
        for i in range(n):
            rv[i] = self.encode_one_img((255 * datast[i][0].view(-1)).int())
            labels[i] = datast[i][1]
        return rv, labels



class ManhattanEncoder(LinearEncoder):
    def __init__(self, dim=10000, num=256, r=2):
        super().__init__(dim=dim, num=num)
        self.r = r
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.item_mem = None

    def compute_modular_distance(self, u, v):
        """Compute the modular Manhattan distance between two vectors."""
        diff1 = torch.remainder(u - v, self.r)
        diff2 = torch.remainder(v - u, self.r)
        delta = torch.minimum(diff1, diff2)
        return torch.sum(delta)

    def group_bundle(self, lst):
        """
        Bundle a list of vectors using modular arithmetic.
        
        Args:
            lst (Tensor): A list of vectors to bundle, shape (num_vectors, dim).
        
        Returns:
            Tensor: A single vector representing the bundled result, shape (dim,).
        """
        # Step 1: Sum all vectors element-wise
        summed_vector = torch.sum(lst, dim=0)
        
        # Step 2: Apply modular arithmetic to ensure all elements are within the range [0, r)
        bundled_vector = torch.remainder(summed_vector, self.r)
        
        return bundled_vector

    def encode_one_img(self, x):
        """Encoding for a single image with modular arithmetic."""
        rv = self.item_mem[x[0]]
        for i in range(1, x.shape[0]):
            rv = torch.roll(rv, i)
            rv = torch.remainder(-rv * self.item_mem[x[i]], self.r)
        return rv

    def encode_data_extract_labels(self, datast):
        n = len(datast)
        rv = torch.zeros((n, self.dim)).to(self.device)
        labels = torch.zeros(n).long().to(self.device)
        print('start encoding data with Manhattan distance...')
        
        for i in range(n):
            # Encode single image directly
            rv[i] = self.encode_one_img((255 * datast[i][0].view(-1)).int())
            labels[i] = datast[i][1]
            
            if (i % 1000 == 999):
                print(f"{i + 1} images encoded")
    
        print('finish encoding data with Manhattan distance')
        return rv, labels
      

    def similarity(self, x, y):
        """
        Compute similarity between vectors using modular Manhattan distance.
        Lower distance means higher similarity.
        """
        n = x.size(-1)
        distance = self.compute_modular_distance(x, y)
        scaled_distance = distance * 4 / (n * self.r)
        return 1 - scaled_distance




class RandomFourierEncoder:
    def __init__(self, input_dim, gamma, gorder=2, output_dim=10000):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.gorder = gorder
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def pts_map(self, x, r=1.0):
        theta = 2.0 * np.pi / (1.0 * self.gorder) * x
        pts = r * torch.stack([torch.cos(theta), torch.sin(theta)], -1)
        return pts

    def GroupRFF(self, x, sigma):
        intervals = sigma * torch.tensor(
            [scipy.stats.norm.ppf(i * 1.0 / self.gorder) for i in range(1, self.gorder)]).float()
        print('the threshold to discretize fourier features to group elements', intervals)
        group_index = torch.zeros_like(x)
        group_index[x <= intervals[0]] = 0
        group_index[x > intervals[-1]] = self.gorder - 1
        if self.gorder > 2:
            for i in range(1, self.gorder - 1):
                group_index[(x > intervals[i - 1]) & (x <= intervals[i])] = i
        return group_index

    def build_item_mem(self):
        correction_factor = 1 / 1.4
        x = np.linspace(0, 255, num=256)
        Cov = np.array([np.exp(-correction_factor * self.gamma ** 2 * ((x - y) / 255.0) ** 2 / 2) for y in range(256)])
        k = Cov.shape[0]
        assert Cov.shape[1] == k, "Cov is not a square matrix."
        L = np.sin(Cov * np.pi / 2.0)
        eigen_values, eigen_vectors = np.linalg.eigh(L)
        R = eigen_vectors @ np.diag(np.maximum(0, eigen_values) ** 0.5) @ eigen_vectors.T
        item_mem = torch.from_numpy(np.random.randn(self.output_dim, k) @ R).float()
        self.item_mem = self.GroupRFF(item_mem, np.sqrt((R ** 2).sum(0).max())).T
        self.item_mem = self.item_mem.to(self.device)
        return self.item_mem

    def encode_one_img(self, x):
        x = x.to(self.device).long()
        bs, channels, num_pixels = x.size()
        rv = self.item_mem[x.flatten()].view(bs, channels, num_pixels, -1).transpose(0, 2)
        for i in range(num_pixels):
            rv[i] = torch.roll(rv[i], shifts=783 - i, dims=-1)
        rv = torch.sum(rv, dim=0)
        if self.gorder == 2:
            rv = rv % 2
        return rv.transpose(0, 1).reshape((bs, -1))

    def encode_data_extract_labels(self, datast):
        channels = datast[0][0].size(0)
        n = len(datast)
        rv = torch.zeros((n, channels * self.output_dim))
        labels = torch.zeros(n).long()
        print('Start encoding data')
        start_time = time.time()
        batch_size = 128
        data_loader = torch.utils.data.DataLoader(datast, batch_size=batch_size, shuffle=False)
        for i, batch_img in enumerate(data_loader):
            num_imgs = batch_img[0].size(0)
            rv[i * batch_size: i * batch_size + num_imgs] = self.encode_one_img(
                (255 * batch_img[0].view(num_imgs, channels, -1)).int())
            labels[i * batch_size: i * batch_size + num_imgs] = batch_img[1]
            if i % 100 == 99:
                print(f"{(i + 1) * batch_size} images encoded. Total time elapse = {time.time() - start_time}")
        print('Finish encoding data')
        return rv, labels

    def group_bind(self, lst):
        results = torch.sum(lst, dim=0)
        return results

    def group_bundle(self, lst):
        intervals = torch.tensor([2 * np.pi / self.gorder * i for i in range(self.gorder)]) + np.pi / self.gorder
        pts = torch.sum(self.pts_map(lst), dim=0)
        raw_angles = 2 * np.pi + torch.arctan(pts[:, 1] / pts[:, 0]) - np.pi * (pts[:, 0] < 0).float()
        angles = torch.fmod(raw_angles, 2 * np.pi)
        return torch.floor(angles / (2.0 * np.pi) * self.gorder + 1 / 2)

    def similarity(self, x, y):
        return torch.sum(torch.sum(self.pts_map(x) * self.pts_map(y), dim=-1), dim=-1) * (1.0 / x.size(-1))
