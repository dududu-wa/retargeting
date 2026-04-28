import argparse
import os
from pathlib import Path

import imageio
import mujoco as mj
import numpy as np

from general_motion_retargeting.params import ROBOT_BASE_DICT, ROBOT_XML_DICT


def load_npz_motion(npz_file: str):
    data = np.load(npz_file, allow_pickle=True)

    if "root_pos" not in data:
        raise KeyError(f"{npz_file} missing key: root_pos")

    if "root_rot_wxyz" in data:
        root_rot = data["root_rot_wxyz"]
    elif "root_rot" in data:
        root_rot = data["root_rot"]
    else:
        raise KeyError(f"{npz_file} missing key: root_rot_wxyz/root_rot")

    if "dof_pos" in data:
        dof_pos = data["dof_pos"]
    elif "dof_positions" in data:
        dof_pos = data["dof_positions"]
    else:
        raise KeyError(f"{npz_file} missing key: dof_pos/dof_positions")

    fps = float(data["fps"][0]) if "fps" in data else 30.0

    root_pos = np.asarray(data["root_pos"], dtype=np.float32)
    root_rot = np.asarray(root_rot, dtype=np.float32)
    dof_pos = np.asarray(dof_pos, dtype=np.float32)

    if not (len(root_pos) == len(root_rot) == len(dof_pos)):
        raise ValueError(
            f"Mismatched frame count: root_pos={len(root_pos)}, root_rot={len(root_rot)}, dof_pos={len(dof_pos)}"
        )

    return root_pos, root_rot, dof_pos, fps


def configure_camera(cam: mj.MjvCamera, distance: float, elevation: float, azimuth: float):
    mj.mjv_defaultCamera(cam)
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.distance = distance
    cam.elevation = elevation
    cam.azimuth = azimuth


def main():
    parser = argparse.ArgumentParser(description="Render a robot motion .npz into an MP4 video without opening a viewer window.")
    parser.add_argument("--npz_file", type=str, required=True)
    parser.add_argument("--robot", type=str, default="r2v2", choices=sorted(list(ROBOT_XML_DICT.keys())))
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--fps", type=float, default=None, help="Override output FPS. Default uses npz fps")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera_distance", type=float, default=2.5)
    parser.add_argument("--camera_elevation", type=float, default=-12.0)
    parser.add_argument("--camera_azimuth", type=float, default=145.0)
    parser.add_argument("--follow_base", action="store_true", default=True)
    parser.add_argument("--gl_backend", type=str, default="egl", choices=["egl", "osmesa", "glfw"])
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", args.gl_backend)

    root_pos, root_rot, dof_pos, motion_fps = load_npz_motion(args.npz_file)
    output_fps = args.fps if args.fps is not None else motion_fps

    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
    data = mj.MjData(model)

    expected_dof = model.nq - 7
    if dof_pos.shape[1] != expected_dof:
        raise ValueError(
            f"DoF mismatch for robot={args.robot}: npz dof={dof_pos.shape[1]}, model dof={expected_dof}."
        )

    video_path = Path(args.video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    cam = mj.MjvCamera()
    configure_camera(cam, args.camera_distance, args.camera_elevation, args.camera_azimuth)

    fb_width = int(model.vis.global_.offwidth)
    fb_height = int(model.vis.global_.offheight)
    render_width = min(args.width, fb_width)
    render_height = min(args.height, fb_height)
    if render_width != args.width or render_height != args.height:
        print(
            f"[offscreen] requested {args.width}x{args.height}, clamped to {render_width}x{render_height} "
            f"(model offscreen buffer: {fb_width}x{fb_height})"
        )

    renderer = mj.Renderer(model, width=render_width, height=render_height)
    writer = imageio.get_writer(str(video_path), fps=output_fps)

    start = max(0, args.start_frame)
    end = min(len(root_pos), start + max(1, args.max_frames))

    base_body_name = ROBOT_BASE_DICT[args.robot]
    base_body_id = model.body(base_body_name).id

    print(f"[offscreen] npz={args.npz_file}")
    print(f"[offscreen] robot={args.robot}, frames={len(root_pos)}, render_range=[{start}, {end})")
    print(f"[offscreen] output={video_path}, fps={output_fps}")

    try:
        for i in range(start, end):
            data.qpos[:3] = root_pos[i]
            data.qpos[3:7] = root_rot[i]
            data.qpos[7:] = dof_pos[i]
            mj.mj_forward(model, data)

            if args.follow_base:
                cam.lookat[:] = data.xpos[base_body_id]

            renderer.update_scene(data, camera=cam)
            frame = renderer.render()
            writer.append_data(frame)

            if (i - start + 1) % 100 == 0:
                print(f"[offscreen] rendered {i - start + 1} frames")
    finally:
        writer.close()
        renderer.close()

    print(f"[offscreen] saved video: {video_path}")


if __name__ == "__main__":
    main()
