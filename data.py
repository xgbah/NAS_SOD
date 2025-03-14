import json
import random
import os
import shutil

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image
from tqdm import tqdm


import torch
from torch.utils.data import Dataset


class dataset(Dataset):
    def __init__(self, path, mode):
        self.all_files = sorted(self.get_all_files(path))
        self.datas = {}
        for file_path in tqdm(self.all_files):
            base_name = os.path.basename(file_path)
            name_without_extension = base_name.split(".")[0]
            if name_without_extension not in self.datas.keys():
                self.datas.update({name_without_extension: {}})
            if "masks" in file_path:
                self.datas[name_without_extension].update({"ground_truth": file_path})
            if "images" in file_path:
                self.datas[name_without_extension].update({"image": file_path})
            if "depth" in file_path:
                self.datas[name_without_extension].update({"depth": file_path})
            if "segments" in file_path:
                self.datas[name_without_extension].update({"segments": file_path})
        if mode == "train":
            # self.train_datas = sorted(list(self.datas.keys()))[:len(self.datas) // 2]
            # self.test_datas = sorted(list(self.datas.keys()))[len(self.datas) // 2:]
            self.train_datas = sorted(list(self.datas.keys()))
            self.test_datas = sorted(list(self.datas.keys()))
        else:
            self.train_datas = sorted(list(self.datas.keys()))
            self.test_datas = sorted(list(self.datas.keys()))

    def get_all_files(self, paths):
        image_paths = []
        for path in paths:
            sub_dirs = [os.path.join(path, i) for i in os.listdir(path)]
            for sub_dir in tqdm(sub_dirs):
                if os.path.isdir(sub_dir):
                    image_paths_ = self.get_all_files([sub_dir])
                    image_paths += image_paths_
                else:
                    image_paths.append(sub_dir)
        return image_paths

    def __len__(self):
        return len(self.train_datas)

    def random_clip(self, x, mode):
        h, w, c = x.shape
        if mode == "global":
            min_h = int(0.35 * h)
            min_w = int(0.35 * w)
            max_h = int(0.65 * h)
            max_w = int(0.65 * w)
        else:
            min_h = int(0.5 * h)
            min_w = int(0.5 * w)
            max_h = int(0.99 * h)
            max_w = int(0.99 * w)
        clip_h = random.randint(min_h, max_h)
        clip_w = random.randint(min_w, max_w)

        slide_h = random.randint(0, h - max_h)
        slide_w = random.randint(0, w - max_w)

        x = x[slide_h:slide_h+clip_h, slide_w:slide_w+clip_w, :]

        if random.random() < 0.5:
            x = x[::-1, :, :]
        if random.random() < 0.5:
            x = x[:, ::-1, :]
        if random.random() < 0.5:
            x = x[:, :, ::-1]
        return x

    def __getitem__(self, index):
        img_size = [384, 384]
        small_size = [384, 384]
        base_name = self.train_datas[index]
        tgt = cv.imread(self.datas[base_name]["ground_truth"]) / 255.0
        tgt = cv.resize(tgt, img_size)

        image = cv.imread(self.datas[base_name]["image"]) / 255.0 * 2 - 1
        image = cv.resize(image, img_size)
        # local_images = [cv.resize(self.random_clip(image, mode="local"), small_size) for _ in range(1)]
        # global_images = [cv.resize(self.random_clip(image, mode="global"), img_size) for _ in range(1)]

        depth = cv.imread(self.datas[base_name]["depth"]) / 255.0 * 2 - 1
        depth = depth if random.random() < 0.5 else -1 * depth
        depth = cv.resize(depth, img_size)

        seg_masks = torch.load(self.datas[base_name]["segments"])
        seg_masks = np.array(seg_masks)
        h, w = seg_masks.shape[-2:]
        seg = np.ones([h, w, 3]) * np.random.random(3) * 255
        for mask in seg_masks[:-1]:
            mask = mask[..., None]
            seg = np.where(mask == 0, seg, mask * np.random.random(3) * 255)
        seg = cv.resize(seg, img_size) / 255.0 * 2 - 1

        # edges = cv.imread(self.datas[base_name]["ground_truth"])
        # edges = cv.resize(edges, img_size)
        # edges = cv.Canny(edges, 100, 200) / 255.0
        train_data = {
            # "image_x": local_images,
            # "image_y": global_images,
            "image": image,
            "depth": depth,
            "seg": seg,
            "tgt": tgt,
            # "edge": edges,
        }
        return train_data


if __name__ == "__main__":
    pass