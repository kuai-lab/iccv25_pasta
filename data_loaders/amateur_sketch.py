import torch
from torch.utils.data import Dataset
from glob import glob
import os
from PIL import Image
import numpy as np
import trimesh

from data_loaders import augment_clipcenter


class AmateurSketchDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.all_files = sorted(glob(self.data_root + '/*'))
        self.all_sketches = sorted(glob(self.data_root + '/*/*.png'))
        self.llava_base_dir = '/home/cvlab/chair_llava_feat/shapenet_amateur' # AmateurSketchDataset llava_base_dir


    def __len__(self):
        return len(self.all_sketches)

    def __getitem__(self, idx):
        sketch_path = self.all_sketches[idx]
        img = Image.open(sketch_path).convert('RGB')
        img = np.array(img)
        sketch = augment_clipcenter.augment_cropped_square(img, 256)

        filename = sketch_path.split('/')[-2]
        num_view = sketch_path.split('/')[-1].split('.')[0]

        max_token_len = 53
        obj_name = filename
        
        llava_feat_path = f'{self.llava_base_dir}/{obj_name}/{num_view}.npy'

        try:
            l_feat = np.load(llava_feat_path)
            l_feat = torch.from_numpy(l_feat)
            
            if l_feat.shape[0] < max_token_len:
                rest = max_token_len - l_feat.shape[0]
                pad = torch.zeros([rest, l_feat.shape[1]])
                l_feat = torch.cat([l_feat, pad], dim=0)
        except:
            l_feat = torch.zeros([max_token_len, 4096])

        mesh_path = os.path.join(self.data_root, filename, 'model.obj')
        mesh = trimesh.load(mesh_path, force='mesh')

        return {
            'filename': filename,
            'num_view': num_view,
            'sketch': sketch,
            'mesh': mesh,
            'l_feat': l_feat
        }


if __name__=='__main__':
    ds = AmateurSketchDataset('/data/shapenet_amateur')

