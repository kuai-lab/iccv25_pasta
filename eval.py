from custom_types import *
import options
from utils import files_utils
from utils.utils import scale_to_unit_sphere
from ui_sketch import sketch_inference
from data_loaders import augment_clipcenter, amateur_sketch
import trimesh
import numpy as np
from tqdm import tqdm
import point_cloud_utils as pcu


def sketch2mesh(path: str, model):
    sketch = files_utils.load_image(path)
    sketch = augment_clipcenter.augment_cropped_square(sketch, 256)
    gmm, mesh, zh_0 = model.sketch2mesh(sketch, get_zh=True)
    return gmm, mesh, zh_0, sketch


def mesh2points(mesh, n_points):
    points = mesh.sample(n_points)
    points = torch.tensor(points, dtype=torch.float32).cuda()
    points = points.unsqueeze(0)
    return points

def rotate_mesh(mesh, angle):
    mesh.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 1, 0]))
    return mesh


def main(tag, spaghetti_tag):
    test_set_path = './assets/data/shapenet_amateur'
    ds = amateur_sketch.AmateurSketchDataset(test_set_path)

    opt = options.SketchOptions(tag=tag, spaghetti_tag=spaghetti_tag)
    model = sketch_inference.SketchInference(opt)
    

    n_points = 100000

    all_chamfer_distances = []
    curr_cd = 0
    for idx, batch in enumerate(tqdm(ds)):
        with torch.no_grad():
            sketch = batch['sketch']
            gt_mesh = batch['mesh']
            l_feat = batch['l_feat'].float().cuda()
            l_feat = l_feat.unsqueeze(0)
            gt_mesh = scale_to_unit_sphere(gt_mesh)
            gt_mesh = rotate_mesh(gt_mesh, np.pi / 2)
            gmm, mesh, zh_0 = model.sketch2mesh(sketch, get_zh=True, l_feat=l_feat)

        pred_mesh = trimesh.Trimesh(vertices=mesh[0], faces=mesh[1])
        pred_mesh = scale_to_unit_sphere(pred_mesh)

        pred_points = mesh2points(pred_mesh, n_points)
        gt_points = mesh2points(gt_mesh, n_points)

        # ================== Chamfer Distance ==================
        cd = pcu.chamfer_distance(pred_points[0].cpu().numpy(), gt_points[0].cpu().numpy())

        all_chamfer_distances.append(cd)
        curr_cd += cd

        # ================= Print =================
        print(f"Chamfer distance: {cd}")
        print(f"Current Mean CD: {curr_cd / (idx + 1)}")
        print("=========================================")


    # Compute the mean of the remaining Chamfer distances
    mean_chamfer_distance = np.mean(all_chamfer_distances)
    print(f"Mean chamfer distance: {mean_chamfer_distance}")
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', dest="tag", action='store', type=str, help='The tag for saving the model', default="test")
    parser.add_argument('--spaghetti_tag', dest="spaghetti_tag", action='store', type=str, help='The tag for spaghetti', default="chairs_sym_hard")
    args = parser.parse_args()
    main(args.tag, args.spaghetti_tag)