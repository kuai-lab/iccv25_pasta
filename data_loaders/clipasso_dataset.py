from PIL import Image
from custom_types import *
from utils import files_utils, edge_detection, rotation_utils, svg_utils
import options
import constants
import torchvision
import subprocess
import io
import cv2 as cv
import matplotlib.pyplot as plt
from data_loaders import augment_clipcenter
import multiprocessing as mp
import numpy as np
import torch


class ClipassoDs(Dataset):

    def get_zh(self, item: int) -> T:
        return self.zh[item]
    
    def get_adj(self, item: int) -> T:
        return self.dist_adj[item]
    
    def get_part_adj(self, item: int) -> T:
        return self.part_dist_adj[item]

    @staticmethod
    def split_by_range(item: T, range_: T) -> TS:
        out = []
        start = 0
        for i in range(len(range_)):
            out.append(item[start: start + range_[i]])
            start += range_[i]
        return out

    def split_by_view(self, raw_data, view):
        res = 256
        depth_maps, masks = raw_data["depth_maps"], raw_data["masks"]
        supports, ranges = raw_data["supports"], raw_data["ranges"]
        points = raw_data["points"]
        depth_maps, supports, points = map(lambda x: self.split_by_range(x, ranges[view]),
                                           (depth_maps[view], supports[view], points[view]))
        mask_range = ranges[view].roll(1)
        mask_range[0] = res ** 2
        masks = self.split_by_range(masks[view], mask_range)
        return points, depth_maps, masks, supports
    
    def augment_svg(self, svg:svg_utils.SVG):
        width = (self.stroke_width_var[0] + (self.stroke_width_var[1] - self.stroke_width_var[0]) * self.get_random())
        svg.change_width_uniform(width)
        return

    def svg_to_img(self, svg:svg_utils.SVG) -> Image.Image:
        # return img
        image_bytes = svg.render(self.res, self.res)
        decoded = cv.imdecode(np.frombuffer(image_bytes, np.uint8), cv.IMREAD_COLOR)
        image = Image.fromarray(decoded)
        return image.convert("RGB")
    
    def augment_img(self, image: Image.Image):
        if self.augmentation:
            pur = self.get_random()
            image = self.hflip_aug(image)
            if pur < .33:
                image = self.perspective_aug(image)
            elif pur < .66:
                image = self.affine_aug(image)
        image = augment_clipcenter.augment_cropped_square(image, 256)
        image = torch.from_numpy(V(image)).float() / 255.
        image = image.mean(-1).unsqueeze(0)
        return image

    def load_item(self, item: int) -> Tuple[svg_utils.SVG, ...]:
        view = (int) (self.get_random() * 6)
        path = self.paths[view][item]
        folder, filename, type = path
        full_path = ''.join(path)
        svg = svg_utils.SVG(full_path)
        infos = filename.split("_")
        id = int(infos[1])
        # view = infos[3]
        return svg, filename, id, view

    def __getitem__(self, item: int):
        svg, filename, id, view = self.load_item(item)
        if self.augmentation:
            self.augment_svg(svg)
        image = self.svg_to_img(svg)
        zh = self.get_zh(id)
        dist_adj = self.get_adj(id)
        part_dist_adj = self.get_part_adj(id)
        image = self.augment_img(image)
        
        
        max_token_len = 53
        obj_name = id
        llava_feat_path = f'{self.llava_base_dir}/view_{view}/{int(obj_name):06d}.npy'

        try:
            l_feat = np.load(llava_feat_path)
            l_feat = torch.from_numpy(l_feat)
            
            if l_feat.shape[0] < max_token_len:
                rest = max_token_len - l_feat.shape[0]
                pad = torch.zeros([rest, l_feat.shape[1]])
                l_feat = torch.cat([l_feat, pad], dim=0)
        except:
            l_feat = torch.zeros([max_token_len, 4096])
        return image, zh, l_feat, dist_adj, part_dist_adj

    def __len__(self):
        return len(self.paths[0])

    def get_random(self):
        with self.cur_rand.get_lock():  # Acquire the lock before modifying
            self.cur_rand.value += 1
            if self.cur_rand.value >= self.size_random:
                self.random_array = np.random.rand(self.size_random)
                self.cur_rand.value = 0
            return self.random_array[self.cur_rand.value]

    '''At the end, self.svgs[i] has the svg corresponding to self.paths[i]'''
    def _load_svgs(self):
        self.svgs = []
        for paths in self.paths:
            for path in paths:
                full_path = ''.join(path)
                svg = svg_utils.SVG(full_path)
                self.svgs.append(svg)


    def set_augmentation(self, augment : bool):
        self.augmentation = augment
    
    def __init__(self, opt: options.SketchOptions):
        self.opt = opt
        self.res = 256
        out_root = f'{constants.DATA_ROOT}/dataset_chair_preprocess/'
        svg_paths = [f'{out_root}/{opt.data_tag}/svg/view_{i}/' for i in range(6)]
        self.paths = [files_utils.collect(svg_path, '.svg') for svg_path in svg_paths]
        self.zh = torch.from_numpy(files_utils.load_np(f'{out_root}zh_0'))
        self.affine_aug = torchvision.transforms.RandomAffine(6, translate=(.078, .078), scale=(.95, 1.05),
                                                                shear=3, fill=(255, 255, 255),
                                                                interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        self.perspective_aug = torchvision.transforms.RandomPerspective(distortion_scale=0.4, p=1.,fill=(255, 255, 255),
                                                                        interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        self.hflip_aug = torchvision.transforms.RandomHorizontalFlip(p=.5)
        self.stroke_width_var = (1.0, 4.5)
        self.augmentation = True
        self.size_random = 10000
        self.random_array = np.random.rand(self.size_random)
        self.cur_rand = mp.Value('i', -1)
        self.llava_base_dir = '/home/cvlab/PASTA_llava/chair_llava_feat/clipasso' # clipasso llava_base_dir

        # load dist adj
        dist_adj_path = f'{constants.DATA_ROOT}dataset_chair_preprocess/chairs_mu_distances.npy'
        self.dist_adj = torch.from_numpy(np.load(dist_adj_path))

        # load part dist adj
        part_dist_adj_path = f'{constants.DATA_ROOT}dataset_chair_preprocess/chairs_mu_distances_part.npy'
        self.part_dist_adj = torch.from_numpy(np.load(part_dist_adj_path))


def render_paths(input_path_svgs, output_path, res=256):
    for path in input_path_svgs:
        path_full = ''.join(path)
        svg = svg_utils.SVG(path_full)
        image_bytes = svg.render(res, res)
        decoded = cv.imdecode(np.frombuffer(image_bytes, np.uint8), cv.IMREAD_COLOR)
        image = Image.fromarray(decoded)
        image = image.convert("RGB")
        image.save(output_path + path_full.split("/")[-1].split(".")[0].split("_")[1] + ".png")
