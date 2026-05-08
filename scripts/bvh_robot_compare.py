"""
Render side-by-side comparison of raw BVH human skeleton vs r2v2 robot motion.
Produces:
  - output/bvh_human_jumps1.mp4   : stick-figure video of raw BVH
  - output/r2v2_robot_jumps1.mp4  : robot video from npz
  - output/compare_jumps1/        : N side-by-side PNG snapshots at random frames
"""

import os
import sys
import argparse
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import imageio
import mujoco as mj

os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, str(Path(__file__).parent.parent))
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.params import ROBOT_XML_DICT, ROBOT_BASE_DICT

# ── skeleton connectivity (LAFAN1 joint names) ──────────────────────────────
SKELETON_EDGES = [
    ("Hips", "Spine"),
    ("Spine", "Spine1"),
    ("Spine1", "Spine2"),
    ("Spine2", "Neck"),
    ("Neck", "Head"),
    # left arm
    ("Spine2", "LeftShoulder"),
    ("LeftShoulder", "LeftArm"),
    ("LeftArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),
    # right arm
    ("Spine2", "RightShoulder"),
    ("RightShoulder", "RightArm"),
    ("RightArm", "RightForeArm"),
    ("RightForeArm", "RightHand"),
    # left leg
    ("Hips", "LeftUpLeg"),
    ("LeftUpLeg", "LeftLeg"),
    ("LeftLeg", "LeftFoot"),
    ("LeftFoot", "LeftToe"),
    # right leg
    ("Hips", "RightUpLeg"),
    ("RightUpLeg", "RightLeg"),
    ("RightLeg", "RightFoot"),
    ("RightFoot", "RightToe"),
]


def render_human_frame(frame: dict, ax: plt.Axes, title: str = ""):
    """Draw stick figure on a 3D matplotlib axes."""
    ax.cla()
    positions = {name: np.array(v[0]) for name, v in frame.items()}

    # draw bones
    for a, b in SKELETON_EDGES:
        if a in positions and b in positions:
            pa, pb = positions[a], positions[b]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                    "b-", linewidth=2)

    # draw joints
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    zs = [p[2] for p in positions.values()]
    ax.scatter(xs, ys, zs, c="red", s=20, zorder=5)

    # fix axes around the hip centre
    hip = positions.get("Hips", np.array([0, 0, 1]))
    r = 1.2
    ax.set_xlim(hip[0] - r, hip[0] + r)
    ax.set_ylim(hip[1] - r, hip[1] + r)
    ax.set_zlim(0, hip[2] + r)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title, fontsize=9)
    ax.view_init(elev=10, azim=135)


def fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[..., :3]
    return img


def render_human_video(bvh_frames, output_path: str, fps: float = 30.0,
                       width: int = 640, height: int = 480):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    writer = imageio.get_writer(output_path, fps=fps)
    for i, frame in enumerate(bvh_frames):
        render_human_frame(frame, ax, title=f"BVH Human  frame {i}")
        img = fig_to_rgb(fig)
        writer.append_data(img)
        if (i + 1) % 200 == 0:
            print(f"  [human] rendered {i+1}/{len(bvh_frames)} frames")
    writer.close()
    plt.close(fig)
    print(f"  [human] saved → {output_path}")


def render_robot_video(npz_file: str, robot: str, output_path: str,
                       width: int = 640, height: int = 480):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data_npz = np.load(npz_file, allow_pickle=True)
    root_pos = np.asarray(data_npz["root_pos"], dtype=np.float32)
    root_rot = np.asarray(data_npz["root_rot_wxyz"], dtype=np.float32)
    dof_pos  = np.asarray(data_npz["dof_pos"],  dtype=np.float32)
    fps      = float(data_npz["fps"][0]) if "fps" in data_npz else 30.0

    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot]))
    sim   = mj.MjData(model)

    cam = mj.MjvCamera()
    mj.mjv_defaultCamera(cam)
    cam.type      = mj.mjtCamera.mjCAMERA_FREE
    cam.distance  = 3.0
    cam.elevation = -15.0
    cam.azimuth   = 145.0

    base_id = model.body(ROBOT_BASE_DICT[robot]).id

    fb_w = int(model.vis.global_.offwidth)
    fb_h = int(model.vis.global_.offheight)
    rw = min(width,  fb_w)
    rh = min(height, fb_h)

    renderer = mj.Renderer(model, width=rw, height=rh)
    writer   = imageio.get_writer(output_path, fps=fps)

    for i in range(len(root_pos)):
        sim.qpos[:3]  = root_pos[i]
        sim.qpos[3:7] = root_rot[i]
        sim.qpos[7:]  = dof_pos[i]
        mj.mj_forward(model, sim)
        cam.lookat[:] = sim.xpos[base_id]
        renderer.update_scene(sim, camera=cam)
        writer.append_data(renderer.render())
        if (i + 1) % 200 == 0:
            print(f"  [robot] rendered {i+1}/{len(root_pos)} frames")

    writer.close()
    renderer.close()
    print(f"  [robot] saved → {output_path}")
    return fps


def capture_human_frame_img(frame: dict, width: int, height: int) -> np.ndarray:
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax  = fig.add_subplot(111, projection="3d")
    render_human_frame(frame, ax)
    img = fig_to_rgb(fig)
    plt.close(fig)
    return img


def capture_robot_frame_img(model, sim, root_pos, root_rot, dof_pos,
                             cam, renderer, base_id) -> np.ndarray:
    sim.qpos[:3]  = root_pos
    sim.qpos[3:7] = root_rot
    sim.qpos[7:]  = dof_pos
    mj.mj_forward(model, sim)
    cam.lookat[:] = sim.xpos[base_id]
    renderer.update_scene(sim, camera=cam)
    return renderer.render().copy()


def make_comparison_snapshots(bvh_frames, npz_file: str, robot: str,
                               out_dir: str, n_snapshots: int = 6,
                               width: int = 640, height: int = 480):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    data_npz = np.load(npz_file, allow_pickle=True)
    root_pos = np.asarray(data_npz["root_pos"], dtype=np.float32)
    root_rot = np.asarray(data_npz["root_rot_wxyz"], dtype=np.float32)
    dof_pos  = np.asarray(data_npz["dof_pos"],  dtype=np.float32)

    n_frames = min(len(bvh_frames), len(root_pos))
    indices  = sorted(random.sample(range(n_frames), min(n_snapshots, n_frames)))
    print(f"  [compare] snapshot frames: {indices}")

    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot]))
    sim   = mj.MjData(model)

    cam = mj.MjvCamera()
    mj.mjv_defaultCamera(cam)
    cam.type      = mj.mjtCamera.mjCAMERA_FREE
    cam.distance  = 3.0
    cam.elevation = -15.0
    cam.azimuth   = 145.0

    base_id  = model.body(ROBOT_BASE_DICT[robot]).id
    fb_w     = int(model.vis.global_.offwidth)
    fb_h     = int(model.vis.global_.offheight)
    rw       = min(width,  fb_w)
    rh       = min(height, fb_h)
    renderer = mj.Renderer(model, width=rw, height=rh)

    for idx in indices:
        human_img = capture_human_frame_img(bvh_frames[idx], rw, rh)
        robot_img = capture_robot_frame_img(
            model, sim,
            root_pos[idx], root_rot[idx], dof_pos[idx],
            cam, renderer, base_id,
        )

        # side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(rw * 2 / 100, rh / 100), dpi=100)
        axes[0].imshow(human_img)
        axes[0].set_title(f"BVH Human  (frame {idx})", fontsize=10)
        axes[0].axis("off")
        axes[1].imshow(robot_img)
        axes[1].set_title(f"R2V2 Robot (frame {idx})", fontsize=10)
        axes[1].axis("off")
        fig.tight_layout()

        out_path = Path(out_dir) / f"compare_frame_{idx:05d}.png"
        fig.savefig(str(out_path), dpi=100)
        plt.close(fig)
        print(f"  [compare] saved {out_path}")

    renderer.close()


def main():
    parser = argparse.ArgumentParser(description="BVH vs robot side-by-side comparison")
    parser.add_argument("--bvh_file",  default="lafan1_bvh/jumps1_subject1.bvh")
    parser.add_argument("--npz_file",  default="lafan1_npz_r2v2/jumps1_subject1.npz")
    parser.add_argument("--robot",     default="r2v2")
    parser.add_argument("--out_dir",   default="output/compare_jumps1")
    parser.add_argument("--human_video",  default="output/bvh_human_jumps1.mp4")
    parser.add_argument("--robot_video",  default="output/r2v2_robot_jumps1.mp4")
    parser.add_argument("--n_snapshots",  type=int, default=6)
    parser.add_argument("--width",  type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Limit frames rendered (for quick testing)")
    parser.add_argument("--skip_videos", action="store_true",
                        help="Skip video rendering, only do snapshots")
    args = parser.parse_args()

    print("Loading BVH …")
    bvh_frames, human_height = load_bvh_file(args.bvh_file, format="lafan1")
    print(f"  {len(bvh_frames)} frames, human height={human_height:.2f}m")

    if args.max_frames:
        bvh_frames = bvh_frames[: args.max_frames]

    if not args.skip_videos:
        print("Rendering human stick-figure video …")
        render_human_video(bvh_frames, args.human_video,
                           width=args.width, height=args.height)

        print("Rendering robot video …")
        render_robot_video(args.npz_file, args.robot, args.robot_video,
                           width=args.width, height=args.height)

    print("Generating comparison snapshots …")
    make_comparison_snapshots(
        bvh_frames, args.npz_file, args.robot,
        out_dir=args.out_dir,
        n_snapshots=args.n_snapshots,
        width=args.width, height=args.height,
    )

    print("Done.")


if __name__ == "__main__":
    main()
