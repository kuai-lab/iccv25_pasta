from PIL import Image
from custom_types import *
from utils import files_utils
import options
import constants
import torchvision
import matplotlib.pyplot as plt
from data_loaders import augment_clipcenter
import random
import multiprocessing as mp

def get_chairs_dict(filename):
    chairs_dict = dict()
    with open(filename, 'r') as file:
        count = 0
        lines = file.readlines()
        for line in lines:
            chairs_dict[line[:-1]] = count
            count+=1
    return chairs_dict

class ProsketchDs(Dataset):

    def get_zh(self, item: int) -> T:
        return self.zh[item]
    
    def get_adj(self, item: int) -> T:
        return self.dist_adj[item]
    
    def get_part_adj(self, item: int) -> T:
        return self.part_dist_adj[item]
    
    def augment_img(self, image):
        image = Image.fromarray(image)
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

    def load_item(self, item: int):
        path = self.paths[item]
        _, filename, _ = path
        file_id, _ = filename.split("-")
        id = self.chairs_dict[file_id]
        image = files_utils.load_image("".join(path))

        llava_feat_path = f'{self.llava_base_dir}/{filename}.npy'

        max_token_len = 53
        try:
            l_feat = np.load(llava_feat_path)
            l_feat = torch.from_numpy(l_feat)
            
            if l_feat.shape[0] < max_token_len:
                rest = max_token_len - l_feat.shape[0]
                pad = torch.zeros([rest, l_feat.shape[1]])
                l_feat = torch.cat([l_feat, pad], dim=0)
        except:
            l_feat = torch.zeros([max_token_len, 4096])
        return image, id, l_feat

    def __getitem__(self, item: int):
        image, id, l_feat = self.load_item(item)
        zh = self.get_zh(id)
        dist_adj = self.get_adj(id)
        part_dist_adj = self.get_part_adj(id)
        image = self.augment_img(image) 
        return image, zh, l_feat, dist_adj, part_dist_adj

    def __len__(self):
        return len(self.paths)
    
    def get_random(self):
        with self.cur_rand.get_lock():  # Acquire the lock before modifying
            self.cur_rand.value += 1
            if self.cur_rand.value >= self.size_random:
                self.random_array = np.random.rand(self.size_random)
                self.cur_rand.value = 0
            return self.random_array[self.cur_rand.value]


    def set_augmentation(self, augment : bool):
        self.augmentation = augment
    
    def __init__(self, opt: options.SketchOptions):
        self.opt = opt
        self.res = 256
        out_root = f'{constants.DATA_ROOT}/dataset_chair_preprocess'
        data_path = f'{out_root}/prosketch/original/'
        self.paths = files_utils.collect(data_path, '.png')
        self.zh = torch.from_numpy(files_utils.load_np(f'{out_root}/zh_0'))
        self.chairs_dict = get_chairs_dict(f'{constants.DATA_ROOT}/dataset_chair_preprocess/chairs_list.txt')
        self.affine_aug = torchvision.transforms.RandomAffine(6, translate=(.078, .078), scale=(.95, 1.05),
                                                                shear=3, fill=(255, 255, 255),
                                                                interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        self.perspective_aug = torchvision.transforms.RandomPerspective(distortion_scale=0.4, p=1.,fill=(255, 255, 255),
                                                                        interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        self.hflip_aug = torchvision.transforms.RandomHorizontalFlip(p=0.5)
        self.size_random = 10000
        self.random_array = np.random.rand(self.size_random)
        self.cur_rand = mp.Value('i', -1)
        self.augmentation = True
        self.llava_base_dir = '/home/cvlab/PASTA_llava/chair_llava_feat/prosketch/original' # prosketch llava_base_dir

        # load dist adj
        dist_adj_path = f'{constants.DATA_ROOT}dataset_chair_preprocess/chairs_mu_distances.npy'
        self.dist_adj = torch.from_numpy(np.load(dist_adj_path))

        # load part dist adj
        part_dist_adj_path = f'{constants.DATA_ROOT}dataset_chair_preprocess/chairs_mu_distances_part.npy'
        self.part_dist_adj = torch.from_numpy(np.load(part_dist_adj_path))


def main(): 
    ds = ProsketchDs(options.SketchOptions(data_tag="prosketch/original/"))
    for _ in range(0, 15):
        j = random.randint(0, len(ds))
        image, _ = ds[j]
        plt.imshow(image[0], cmap='gray')
        plt.axis("off")
        plt.show()
    return 0

if __name__ == '__main__':
    main()
