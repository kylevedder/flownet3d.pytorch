#!/usr/bin/env python
import numpy as np
# Numpy set random seed
np.random.seed(0)

import copy
import argparse
import glob
import open3d as o3d
import multiprocessing
from scipy.spatial.transform import Rotation as R
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('dataset_path', type=str, help='Path to dataset')
args = parser.parse_args()

# Validate that dataset path is a valid directory
if not Path(args.dataset_path).is_dir():
    raise ValueError("Invalid dataset path")
# if dataset path doesn't have a trailing slash, add it
if args.dataset_path[-1] != '/':
    args.dataset_path += '/'


def load_intrinsic_matrix(dataset_path):
    """
    Loads the intrinsic matrix from the given directory.
    """
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsics.width, camera_intrinsics.height = np.load(
        dataset_path + "depth_info_size.npy")
    camera_intrinsics.intrinsic_matrix = np.load(dataset_path +
                                                 "/depth_info_K.npy")
    return camera_intrinsics


def get_rgb_depth_files(dataset_path):
    rgb_files = glob.glob(dataset_path + 'rgb_*.png')
    depth_files = glob.glob(dataset_path + 'depth_*.png')
    return zip(rgb_files, depth_files)


def rgb_depth_files_to_rgbd(rgb_file, depth_file):
    """
    Reads rgb and depth images by loading them with Open3d, and creates 
    an Open3D RGBD image.

    Returns the Open3D RGBD image.
    """
    rgb = o3d.io.read_image(rgb_file)
    depth = o3d.io.read_image(depth_file)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb, depth, depth_trunc=5, convert_rgb_to_intensity=False)
    return rgbd


def rgbd_to_point_cloud(rgbd, camera_intrinsics):
    """
    Converts an Open3D RGBD image to an Open3D PointCloud using the given camera intrinsics.
    """
    return o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, camera_intrinsics)


def add_noise_to_point_cloud(point_cloud,
                             position_noise_std=0.0005,
                             color_noise_std=0.01):
    """
    Adds noise to the points and colors of an Open3D point cloud.

    Converts the point cloud points and colors to numpy arrays and 
    adds noise to the points and colors, and then converts them back to
    Open3D Vector3dVector and assign them to the point cloud.

    Returns: point_cloud
    """
    # Deep copy the point cloud to avoid modifying the original
    point_cloud = copy.deepcopy(point_cloud)
    # Convert point cloud to numpy arrays
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    # Add noise to the points and colors
    points += np.random.normal(0, position_noise_std, points.shape)
    colors += np.random.normal(0, color_noise_std, colors.shape)
    # Convert numpy arrays back to Open3D Vector3dVector
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud


def rotate_point_cloud_region_and_generate_flow(
        input_point_cloud,
        num_regions=30,
        region_center_percentage_of_max=0.95,
        min_region_extent=0.2,
        max_region_extent=0.8):
    """
    Generates random region of point cloud using Open3d's bounding box, 
    randomly rotates the box around an edge, and generates a flow for 
    the point cloud.

    Returns (input point cloud, rotated point cloud, flow)
    """
    # Deep copy the point cloud to avoid modifying the original
    point_cloud = copy.deepcopy(input_point_cloud)

    point_clouds_mean = np.mean(np.asarray(point_cloud.points), axis=0)
    point_cloud_min = np.min(np.asarray(point_cloud.points), axis=0)
    point_cloud_max = np.max(np.asarray(point_cloud.points), axis=0)
    min_center = (point_cloud_min - point_clouds_mean
                  ) * region_center_percentage_of_max + point_clouds_mean
    max_center = (point_cloud_max - point_clouds_mean
                  ) * region_center_percentage_of_max + point_clouds_mean

    def generate_oriented_bounding_box():
        """
        Generates an oriented bounding box with random center between 
        min_center and max_center with random Z rotation.

        Returns the oriented bounding box.
        """
        # Generate random center
        center = np.random.uniform(min_center, max_center)
        # Generate random rotation
        rotation = R.from_euler('y', np.random.uniform(0,
                                                       2 * np.pi)).as_matrix()
        # Generate oriented bounding box
        oriented_bounding_box = o3d.geometry.OrientedBoundingBox(
            center, rotation,
            np.random.uniform(min_region_extent,
                              max_region_extent,
                              size=(3, 1)))
        return oriented_bounding_box

    def box_has_min_points(oriented_bounding_box, point_cloud, min_points=20):
        """
        Checks if the oriented bounding box has the min points in it.

        Returns True if the oriented bounding box has points in it, 
        False otherwise.
        """
        point_idxs = oriented_bounding_box.get_point_indices_within_bounding_box(
            point_cloud.points)
        return len(point_idxs) > min_points

    def boxes_overlap(b1, b2):
        """
        Checks if two oriented bounding boxes overlap.

        Convert the oriented bounding boxes into Open3D triangle meshes and 
        check if those overlap, because Open3D's OrientedBoundinBox class 
        doesn't have an overlap method.

        Returns True if the oriented bounding boxes overlap, False otherwise.
        """
        # Convert oriented bounding boxes to triangle meshes
        mesh1 = o3d.geometry.TriangleMesh.create_box(*b1.extent)
        mesh1.rotate(b1.R)
        mesh1.translate(b1.center)
        mesh2 = o3d.geometry.TriangleMesh.create_box(*b2.extent)
        mesh2.rotate(b2.R)
        mesh2.translate(b2.center)
        # Check if the triangle meshes overlap
        return mesh1.is_intersecting(mesh2)

    def remove_unfit_boxes(random_bounding_boxes, point_cloud):
        """
        Removes random bounding boxes that don't have enough points in them, 
        biased towards larger boxes.

        Returns a list of random bounding boxes.
        """
        # Sort boxes by volume, so that the smaller boxes are removed in the event of
        # an overlap
        random_bounding_boxes.sort(key=lambda bb: bb.volume(), reverse=False)

        # Remove boxes that don't have enough points in them
        random_bounding_boxes = [
            bb for bb in random_bounding_boxes
            if box_has_min_points(bb, point_cloud)
        ]
        should_be_removed = np.zeros(len(random_bounding_boxes), dtype=bool)
        for idx in range(len(random_bounding_boxes)):
            if should_be_removed[idx]:
                continue
            for jdx in range(idx + 1, len(random_bounding_boxes)):
                if should_be_removed[jdx]:
                    continue
                if boxes_overlap(random_bounding_boxes[idx],
                                 random_bounding_boxes[jdx]):
                    should_be_removed[jdx] = True

        return [
            bb for idx, bb in enumerate(random_bounding_boxes)
            if not should_be_removed[idx]
        ]

    def generate_random_rotation_matrix(max_rot_deg=15):
        """
        Generate random rotation matrix along the Z and X axes with at most max_rot_deg rotation.
        """
        return R.from_euler('xz',
                            np.random.uniform(-max_rot_deg,
                                              max_rot_deg,
                                              size=(2, )),
                            degrees=True).as_matrix()

    def apply_rotation_to_point_cloud_region(point_cloud, bounding_box,
                                             rotation_matrix):
        """
        Applies rotation to the point cloud region on a random corner.
        """
        # Select random corner from the bounding box corners as the rotation center
        box_corners = np.asarray(bounding_box.get_box_points())
        rotation_center = box_corners[np.random.randint(0, len(box_corners))]
        # Get point indices within the bounding box
        box_point_idxs = bounding_box.get_point_indices_within_bounding_box(
            point_cloud.points)
        # Get the points within the bounding box
        full_points_np = np.asarray(point_cloud.points)
        box_points_np = full_points_np[box_point_idxs]
        # Center around the rotation center
        box_points_np -= rotation_center
        # Rotate the points
        box_points_np = (rotation_matrix @ box_points_np.T).T
        # Translate the points back to the rotation center
        box_points_np += rotation_center
        # Assign the rotated points and colors to the point cloud
        full_points_np[box_point_idxs] = box_points_np
        point_cloud.points = o3d.utility.Vector3dVector(full_points_np)

    def compute_flow_between_point_clouds(point_cloud_1, point_cloud_2):
        """
        Computes the flow between the two point clouds.

        Convert the points to numpy arrays and compute the vector 
        difference between the points in point_cloud_1 and point_cloud_2.

        Returns an Nx3 numpy array of flow vectors.
        """
        # Convert point cloud to numpy arrays
        points_1 = np.asarray(point_cloud_1.points)
        points_2 = np.asarray(point_cloud_2.points)
        # Compute the vector difference between the points
        flow = points_2 - points_1
        return flow

    # Generate num_regions length list of random bounding boxes to rotate
    random_bounding_boxes = [
        generate_oriented_bounding_box() for _ in range(num_regions)
    ]

    # Remove boxes that overlap or lack points
    random_bounding_boxes = remove_unfit_boxes(random_bounding_boxes,
                                               point_cloud)

    # Generate random rotation matrices
    random_rotation_matrices = [
        generate_random_rotation_matrix()
        for _ in range(len(random_bounding_boxes))
    ]

    # Apply the random rotation matrices to the point cloud regions
    for bb, rotation_matrix in zip(random_bounding_boxes,
                                   random_rotation_matrices):
        apply_rotation_to_point_cloud_region(point_cloud, bb, rotation_matrix)

    # # Visualize the random bounding boxes using Open3d
    # o3d.visualization.draw_geometries(random_bounding_boxes + [point_cloud])

    return input_point_cloud, point_cloud, compute_flow_between_point_clouds(
        input_point_cloud, point_cloud)


def point_clouds_to_line_set(pc1, pc2):
    """
    Converts the given point clouds to an Open3D lineset.
    """
    assert (len(pc1.points) == len(pc2.points))
    lineset = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
        pc1, pc2, [(e, e) for e in range(len(pc1.points))])
    # set lineset to be green
    lineset = lineset.paint_uniform_color([0, 1, 0])
    return lineset


def save_point_cloud_and_flow_as_npz(idx, pc1, pc2, flow):
    """
    Saves the given point cloud and flow as a npz file.
    """
    file = f'{args.dataset_path}/TRAIN_robot_pc_{idx:06d}.npz'
    out_dict = {
        'points1': np.asarray(pc1.points),
        'points2': np.asarray(pc2.points),
        'color1': np.asarray(pc1.colors) / 255,
        'color2': np.asarray(pc2.colors) / 255,
        'flow': flow,
        'valid_mask1': np.ones(flow.shape[:2], dtype=bool)
    }
    np.savez_compressed(file, **out_dict)


def process_rgb_and_depth(val):
    idx, (rgb_file, depth_file) = val
    print(f'Processing {idx:06d}')
    rgbd = rgb_depth_files_to_rgbd(rgb_file, depth_file)
    pc = rgbd_to_point_cloud(rgbd, camera_intrinsics)
    pc_in, pc_out, flow = rotate_point_cloud_region_and_generate_flow(pc)
    save_point_cloud_and_flow_as_npz(idx, pc_in, pc_out, flow)

camera_intrinsics = load_intrinsic_matrix(args.dataset_path)
rgb_and_depth_files = get_rgb_depth_files(args.dataset_path)

# Use multiprocessing to iterate over rgb_and_depth_files and process 
# them with process_rgb_and_depth.
with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    pool.map(process_rgb_and_depth, enumerate(rgb_and_depth_files))
