#!/usr/bin/env python

from flownet3d import FlowNet3D
import torch
import copy
import os
import numpy as np
import glob
import open3d as o3d

root = '/data/flyingthings3d/data_processed_maxcut_35_20k_2k_8192'
datapath = list(glob.glob(os.path.join(root, 'TRAIN*.npz')))
model_path = "models/net_self_trained.pth"


print(f"Loading network {model_path}")
net = FlowNet3D().cuda()
net.load_state_dict(torch.load(model_path))
net.eval()
print(f"Loaded network")

print(f"Found {len(datapath)} datapoints")

def eval_flow(pc1, pc2, resample_times=1):
    with torch.no_grad():

        def get_center(data):
            return np.mean(np.asarray(data, dtype=np.float32).T, 1)

        def pc_data_to_torch(data, center=None):
            if center is None:
                return torch.unsqueeze(
                    torch.from_numpy(np.asarray(data, dtype=np.float32).T),
                    0).contiguous().cuda()
            else:
                return torch.unsqueeze(
                    torch.from_numpy(
                        (np.asarray(data, dtype=np.float32) - center).T),
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
        return (pred_flow_sum.T).cpu().numpy()


def make_origin_sphere():
    m = o3d.geometry.TriangleMesh.create_sphere(0.1)
    m.paint_uniform_color([0, 0, 0])
    return m

def load_file(fn):
    with open(fn, 'rb') as fp:
        data = np.load(fp)
        pos1 = data['points1'].astype('float32')
        pos2 = data['points2'].astype('float32')
        color1 = data['color1'].astype('float32') / 255
        color2 = data['color2'].astype('float32') / 255
        flow = data['flow'].astype('float32')
        mask1 = data['valid_mask1']
    return data, pos1, pos2, color1, color2, flow, mask1

data, pos1, pos2, color1, color2, flow, mask1 = load_file(datapath[11000])
color1 *= 255
color2 *= 255

pos2 = pos1
color2 = color1
flow = np.zeros_like(flow)

def to_vec(vals):
    return o3d.utility.Vector3dVector(vals)

def flow_to_lineset(init_pc, flow, color=(0, 1, 0)):
    if len(flow.shape) == 3:
        flow = flow[:, :, 0]
    flowd_pc = copy.deepcopy(init_pc)
    flowd_pc.points = o3d.utility.Vector3dVector(
        np.asarray(flowd_pc.points) + flow)
    correspondences = [(i, i) for i in range(len(flowd_pc.points))]
    lineset = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
        init_pc, flowd_pc, correspondences)
    lineset = lineset.paint_uniform_color(color)
    return lineset, flowd_pc


print(pos1.shape)
print(color1.shape)
print(flow.shape)

pc1 = o3d.geometry.PointCloud()
pc1.points = to_vec(pos1)
pc1.colors = to_vec(color1)

pc2 = o3d.geometry.PointCloud()
pc2.points = to_vec(pos2)
pc2.colors = to_vec(color2)

est_flow = eval_flow(pc1, pc2)

viewer = o3d.visualization.Visualizer()
viewer.create_window(window_name="PointCloud Viewer")

viewer.add_geometry(make_origin_sphere())


viewer.add_geometry(pc1)
true_lineset, true_flow_pc = flow_to_lineset(pc1, flow)
est_lineset, est_flow_pc = flow_to_lineset(pc1, est_flow, color=(0, 0, 1))
viewer.add_geometry(true_lineset)
viewer.add_geometry(est_lineset)
viewer.add_geometry(true_flow_pc)
viewer.add_geometry(est_flow_pc)


# for idx, pc in enumerate(pcs[15:17]):
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

