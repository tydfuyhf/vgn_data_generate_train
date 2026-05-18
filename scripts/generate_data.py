import argparse
from pathlib import Path

from mpi4py import MPI
import numpy as np
import open3d as o3d
import scipy.signal as signal
from tqdm import tqdm

from vgn.grasp import Grasp, Label
from vgn.io import *
from vgn.perception import *
from vgn.simulation import ClutterRemovalSim
from vgn.utils.transform import Rotation, Transform


OBJECT_COUNT_LAMBDA = 4
MAX_VIEWPOINT_COUNT = 6
GRASPS_PER_SCENE = 120

TOP_THETA_RANGE = (0.0, np.deg2rad(15.0))
TOP_RADIUS_RANGE = (1.8, 2.4)
EE_THETA_RANGE = (np.deg2rad(45.0), np.deg2rad(80.0))
EE_RADIUS_RANGE = (1.2, 2.0)


def main(args):
    workers, rank = setup_mpi()
    sim = ClutterRemovalSim(args.scene, args.object_set, gui=args.sim_gui)
    finger_depth = sim.gripper.finger_depth
    grasps_per_worker = args.num_grasps // workers
    pbar = tqdm(total=grasps_per_worker, disable=rank != 0)

    if rank == 0:
        (args.root / "scenes").mkdir(parents=True, exist_ok=True)
        write_setup(
            args.root,
            sim.size,
            sim.camera.intrinsic,
            sim.gripper.max_opening_width,
            sim.gripper.finger_depth,
        )

    for _ in range(grasps_per_worker // GRASPS_PER_SCENE):
        # generate heap
        object_count = np.random.poisson(OBJECT_COUNT_LAMBDA) + 1
        sim.reset(object_count)
        sim.save_state()

        # render synthetic depth images
        depth_imgs, extrinsics = render_images(sim, args)

        # reconstrct point cloud using a subset of the images
        tsdf = create_tsdf(sim.size, 120, depth_imgs, sim.camera.intrinsic, extrinsics)
        pc = tsdf.get_cloud()

        # crop surface and borders from point cloud
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(sim.lower, sim.upper)
        pc = pc.crop(bounding_box)
        # o3d.visualization.draw_geometries([pc])

        if pc.is_empty():
            print("Point cloud empty, skipping scene")
            continue

        # store the raw data
        scene_id = write_sensor_data(args.root, depth_imgs, extrinsics)

        for _ in range(GRASPS_PER_SCENE):
            # sample and evaluate a grasp point
            point, normal = sample_grasp_point(pc, finger_depth)
            grasp, label = evaluate_grasp_point(sim, point, normal)

            # store the sample
            write_grasp(args.root, scene_id, grasp, label)
            pbar.update()

    pbar.close()


def setup_mpi():
    workers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return workers, rank


def render_images(sim, args):
    if args.view_policy == "paired":
        return render_paired_top_ee_images(sim, args)
    else:
        n = np.random.randint(MAX_VIEWPOINT_COUNT) + 1
        return render_legacy_images(sim, n)


def render_legacy_images(sim, n):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0])

    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), np.float32)

    for i in range(n):
        r = np.random.uniform(1.6, 2.4) * sim.size
        theta = np.random.uniform(0.0, np.pi / 4.0)
        phi = np.random.uniform(0.0, 2.0 * np.pi)

        extrinsic = camera_on_sphere(origin, r, theta, phi)
        depth_img = sim.camera.render(extrinsic)[1]

        extrinsics[i] = extrinsic.to_list()
        depth_imgs[i] = depth_img

    return depth_imgs, extrinsics


def render_paired_top_ee_images(sim, args):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0])

    extrinsics = np.empty((2, 7), np.float32)
    depth_imgs = np.empty((2, height, width), np.float32)

    top_extrinsic = sample_top_like_extrinsic(sim, origin)
    ee_extrinsic = sample_ee_like_extrinsic(sim, origin, args)

    depth_imgs[0] = sim.camera.render(top_extrinsic)[1]
    depth_imgs[1] = sim.camera.render(ee_extrinsic)[1]
    extrinsics[0] = top_extrinsic.to_list()
    extrinsics[1] = ee_extrinsic.to_list()

    return depth_imgs, extrinsics


def sample_top_like_extrinsic(sim, origin):
    r = np.random.uniform(*TOP_RADIUS_RANGE) * sim.size
    theta = np.random.uniform(*TOP_THETA_RANGE)
    phi = np.random.uniform(0.0, 2.0 * np.pi)
    return camera_on_sphere(origin, r, theta, phi)


def sample_ee_like_extrinsic(sim, origin, args):
    r = np.random.uniform(*EE_RADIUS_RANGE) * sim.size
    theta = np.random.uniform(*EE_THETA_RANGE)
    phi = sample_ee_phi(args)
    return camera_on_sphere(origin, r, theta, phi)


def sample_ee_phi(args):
    if args.ee_phi_center_deg is None:
        return np.random.uniform(0.0, 2.0 * np.pi)

    center = np.deg2rad(args.ee_phi_center_deg)
    span = np.deg2rad(args.ee_phi_span_deg)
    return np.random.uniform(center - span, center + span)


def sample_grasp_point(point_cloud, finger_depth, eps=0.1):
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    ok = False
    while not ok:
        # TODO this could result in an infinite loop, though very unlikely
        idx = np.random.randint(len(points))
        point, normal = points[idx], normals[idx]
        ok = normal[2] > -0.1  # make sure the normal is poitning upwards
    grasp_depth = np.random.uniform(-eps * finger_depth, (1.0 + eps) * finger_depth)
    point = point + normal * grasp_depth
    return point, normal


def evaluate_grasp_point(sim, pos, normal, num_rotations=6):
    # define initial grasp frame on object surface
    z_axis = -normal
    x_axis = np.r_[1.0, 0.0, 0.0]
    if np.isclose(np.abs(np.dot(x_axis, z_axis)), 1.0, 1e-4):
        x_axis = np.r_[0.0, 1.0, 0.0]
    y_axis = np.cross(z_axis, x_axis)
    x_axis = np.cross(y_axis, z_axis)
    R = Rotation.from_matrix(np.vstack((x_axis, y_axis, z_axis)).T)

    # try to grasp with different yaw angles
    yaws = np.linspace(0.0, np.pi, num_rotations)
    outcomes, widths = [], []
    for yaw in yaws:
        ori = R * Rotation.from_euler("z", yaw)
        sim.restore_state()
        candidate = Grasp(Transform(ori, pos), width=sim.gripper.max_opening_width)
        outcome, width = sim.execute_grasp(candidate, remove=False)
        outcomes.append(outcome)
        widths.append(width)

    # detect mid-point of widest peak of successful yaw angles
    # TODO currently this does not properly handle periodicity
    successes = (np.asarray(outcomes) == Label.SUCCESS).astype(float)
    if np.sum(successes):
        peaks, properties = signal.find_peaks(
            x=np.r_[0, successes, 0], height=1, width=1
        )
        idx_of_widest_peak = peaks[np.argmax(properties["widths"])] - 1
        ori = R * Rotation.from_euler("z", yaws[idx_of_widest_peak])
        width = widths[idx_of_widest_peak]

    return Grasp(Transform(ori, pos), width), int(np.max(outcomes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="pile")
    parser.add_argument("--object-set", type=str, default="blocks")
    parser.add_argument("--num-grasps", type=int, default=10000)
    parser.add_argument("--sim-gui", action="store_true")
    parser.add_argument(
        "--view-policy",
        type=str,
        choices=["paired", "legacy"],
        default="paired",
        help="paired renders one top-like and one EE-like view per scene; legacy uses the original random top-like views",
    )
    parser.add_argument(
        "--ee-phi-center-deg",
        type=float,
        default=None,
        help="center yaw for EE-like views in degrees; omit for uniform 0..360 sampling",
    )
    parser.add_argument(
        "--ee-phi-span-deg",
        type=float,
        default=90.0,
        help="EE-like yaw is sampled from center +/- this value when --ee-phi-center-deg is set",
    )
    args = parser.parse_args()
    main(args)
