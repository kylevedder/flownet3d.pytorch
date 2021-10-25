#!/usr/bin/env python
from matplotlib import lines
from flownet3d import FlowNet3D

import copy
import torch
import argparse
import glob
import open3d as o3d
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy import stats
import numpy as np

parser = argparse.ArgumentParser(description='Process pointclouds')
parser.add_argument("--input_model",
                    default="models/net_self_trained.pth",
                    help="Input directory.")
parser.add_argument("input_dir", help="Input directory.")
args = parser.parse_args()
assert (Path(args.input_dir).is_dir())


def make_origin_sphere():
    m = o3d.geometry.TriangleMesh.create_sphere(0.1)
    m.paint_uniform_color([0, 0, 0])
    return m


def quat_to_mat(quat):
    return R.from_quat(quat).as_matrix()


def get_odom_delta(next, prior):
    return next @ prior.T


def normalize_start(pos, rot, start_pos, start_rot):
    pos = pos - start_pos
    rot = get_odom_delta(rot, start_rot)
    return pos, rot


def to_homogenious_matrix(pos, rot):
    mat = np.eye(4)
    mat[:3, 3] = pos
    mat[:3, :3] = rot
    return mat


def get_intrinsic_matrix(input_dir):
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsics.width, camera_intrinsics.height = np.load(
        input_dir + "/depth_info_size.npy")
    camera_intrinsics.intrinsic_matrix = np.load(args.input_dir +
                                                 "/depth_info_K.npy")
    return camera_intrinsics


def get_rgbd_odoms(input_dir: str):
    depth_imgs = sorted(glob.glob(input_dir + "/depth_*.png"))
    rgb_imgs = sorted(glob.glob(input_dir + "/rgb_*.png"))
    odom_infos = sorted(glob.glob(input_dir + "/odom_*.npy"))
    # depth_imgs = depth_imgs[max_frames:max_frames*2]
    # rgb_imgs = rgb_imgs[max_frames:max_frames*2]
    # odom_infos = odom_infos[max_frames:max_frames*2]
    assert len(depth_imgs) == len(
        rgb_imgs), f"{len(depth_imgs)} vs {len(rgb_imgs)}"

    

    depth_imgs = [o3d.io.read_image(e) for e in depth_imgs]
    rgb_imgs = [o3d.io.read_image(e) for e in rgb_imgs]
    odom_infos = [np.load(e) for e in odom_infos]
    odom_infos = [(e[1:4], quat_to_mat(e[4:])) for e in odom_infos]
    odom_infos = [normalize_start(*e, *odom_infos[0]) for e in odom_infos]
    odom_infos = [to_homogenious_matrix(*e) for e in odom_infos]

    print(
        f"len(odom_infos) {len(odom_infos)} len(depth_imgs) {len(depth_imgs)} len(rgb_imgs) {len(rgb_imgs)}"
    )

    rgbd_odoms = [
        (
            o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb,
                depth,
                depth_scale=1000,  # Converts from mm to meters.
                depth_trunc=4,  # Truncate the depth image in meters.
                convert_rgb_to_intensity=False),
            odom) for rgb, depth, odom in zip(rgb_imgs, depth_imgs, odom_infos)
    ]

    print(f"Returning {len(rgbd_odoms)} rgbd_odoms")

    return rgbd_odoms


def rgbd_odom_to_pc(rgbd, odom, camera_intrinsics):
    pc = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, camera_intrinsics)
    pc_to_robot_frame = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    pc.rotate(pc_to_robot_frame, [0, 0, 0])
    pc.transform(odom)
    # pc = pc.voxel_down_sample(0.1)
    return pc


print(f"Loading network {args.input_model}")
net = FlowNet3D().cuda()
net.load_state_dict(torch.load(args.input_model))
net.eval()
print(f"Loaded network")


def eval_flow(pc1, pc2, resample_times=1):
    with torch.no_grad():

        def get_center(data):
            return np.mean(np.asarray(data, dtype=np.float32).T, 1)

        def pc_data_to_torch(data, center=None):
            np_arr = np.asarray(data, dtype=np.float32)
            if center is not None:
                 np_arr -= center
            np_arr = np_arr.T
            np_arr = rbt_to_flow_frame @ np_arr
            print(np_arr.shape)
            return torch.unsqueeze(
                    torch.from_numpy(
                        np_arr),
                    0).contiguous().cuda()

        pc1_center = get_center(pc1.points)
        pc1_points = pc_data_to_torch(pc1.points, pc1_center)
        pc2_points = pc_data_to_torch(pc2.points, pc1_center)
        pc1_colors = pc_data_to_torch(pc1.colors) / 255
        pc2_colors = pc_data_to_torch(pc2.colors) / 255

        print(f"pc1_points.shape {pc1_points.shape}")
        print(f"pc2_points.shape {pc2_points.shape}")
        print(f"pc1_colors.shape {pc1_colors.shape}")
        print(f"pc2_colors.shape {pc2_colors.shape}")

        pred_flow_sum = torch.zeros_like(pc1_points).cuda()
        
        # resample
        for _ in range(resample_times):
            perm = torch.randperm(pc1_points.shape[2])
            points1_perm = pc1_points[:, :, perm]
            points2_perm = pc2_points[:, :, perm]
            features1_perm = pc1_colors[:, :, perm]
            features2_perm = pc2_colors[:, :, perm]

            # forward
            pred_flow = net(points1_perm, points2_perm, features1_perm, features2_perm)
            pred_flow_sum[:, :, perm] += pred_flow
            pred_flow_sum=pred_flow_sum
            perm = torch.randperm(pc1_points.shape[2])
            # forward
            pred_flow = net(pc1_points, pc2_points, pc1_colors, pc2_colors)
            pred_flow_sum += pred_flow
        
        # statistics
        pred_flow_sum /= resample_times
        return np.linalg.inv(rbt_to_flow_frame) @ (pred_flow_sum.T).cpu().numpy()


def flow_to_lineset(init_pc, flow):
    assert flow.shape[2] == 1
    flow = flow[:, :, 0]
    print(stats.describe(np.linalg.norm(flow, axis=1)))
    flowd_pc = copy.deepcopy(init_pc)
    flowd_pc.points = o3d.utility.Vector3dVector(
        np.asarray(flowd_pc.points) + flow)
    correspondences = [(i, i) for i in range(len(flowd_pc.points))]
    lineset = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
        init_pc, flowd_pc, correspondences)
    lineset = lineset.paint_uniform_color((0, 1, 0))
    return lineset, flowd_pc

def resize_pc_pairs(pc1, pc2):
    min_size = min(len(pc1.points), len(pc2.points))
    pc1_perm = np.random.permutation(len(pc1.points))
    pc2_perm = np.random.permutation(len(pc2.points))
    
    pc1_points = np.array(pc1.points)[pc1_perm][:min_size]
    pc1_colors = np.array(pc1.colors)[pc1_perm][:min_size]

    pc2_points = np.array(pc2.points)[pc2_perm][:min_size]
    pc2_colors = np.array(pc2.colors)[pc2_perm][:min_size]

    pc1.points = o3d.utility.Vector3dVector(pc1_points)
    pc1.colors = o3d.utility.Vector3dVector(pc1_colors)

    pc2.points = o3d.utility.Vector3dVector(pc2_points)
    pc2.colors = o3d.utility.Vector3dVector(pc2_colors)

    return pc1, pc2

rbt_to_flow_frame = np.array([[0, 1, 0],
                              [0, 0, 1],
                              [1, 0, 0]], dtype=np.float32)
camera_intrinsics = get_intrinsic_matrix(args.input_dir)
rgbd_odoms = get_rgbd_odoms(args.input_dir)
pcs = [
    rgbd_odom_to_pc(rgbd, odom, camera_intrinsics) for rgbd, odom in rgbd_odoms
]
pc_pairs = list(zip(pcs, pcs[1:]))
pc_pairs = [resize_pc_pairs(*e) for e in pc_pairs]

model_pc = pc_pairs[200]


# o3d.utility.set_verbosity_level(o3d.utility.Debug)
viewer = o3d.visualization.Visualizer()
viewer.create_window(window_name="PointCloud Viewer")

viewer.add_geometry(make_origin_sphere())

flow = eval_flow(*model_pc)
lineset, flowd_pc = flow_to_lineset(model_pc[0], flow)
print(f"Flow shape: {flow.shape}")

viewer.add_geometry(model_pc[0])
viewer.add_geometry(model_pc[1])
viewer.add_geometry(lineset)

# for idx, pc in enumerate(pcs):
#     viewer.add_geometry(pc)

view = viewer.get_view_control()
view.set_lookat([0, 0, 0])
view.set_front([-1, 0, 0])
view.set_up([0, 0, 1])
opt = viewer.get_render_option()
opt.show_coordinate_frame = True
opt.background_color = np.asarray([0.5, 0.5, 0.5])
opt.point_size = 0.01
viewer.run()
viewer.destroy_window()
