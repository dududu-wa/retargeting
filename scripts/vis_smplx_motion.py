"""
可视化 SMPLX 人体 mesh 动作。
用法:
  python scripts/vis_smplx_motion.py --smplx_file data/amass/HDM05/bk/HDM_bk_01-01_01_120_stageii.npz
"""
import argparse
import pathlib
import time
import numpy as np
import torch
import smplx
import trimesh
import trimesh.viewer

from general_motion_retargeting.utils.smpl import load_smplx_file


def main():
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser(description="可视化 SMPLX 人体 mesh 动作")
    parser.add_argument(
        "--smplx_file",
        type=str,
        required=True,
        help="SMPLX motion file (.npz)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30,
        help="播放帧率",
    )
    args = parser.parse_args()

    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"

    # 加载 SMPLX 数据
    smplx_data, body_model, smplx_output, human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )

    # 获取顶点和面
    vertices_all = smplx_output.vertices.detach().numpy()  # (N, V, 3)
    faces = body_model.faces  # (F, 3)

    src_fps = smplx_data["mocap_frame_rate"].item()
    frame_skip = max(1, int(src_fps / args.fps))
    num_frames = vertices_all.shape[0]

    print(f"SMPLX 文件: {args.smplx_file}")
    print(f"总帧数: {num_frames}, 原始帧率: {src_fps}, 播放帧率: {args.fps}")
    print(f"人体身高估计: {human_height:.2f}m")
    print(f"顶点数: {vertices_all.shape[1]}, 面数: {faces.shape[0]}")
    print()
    print("正在启动可视化... (关闭窗口退出)")

    # 用 pyrender 或 trimesh 的 scene viewer
    # 由于 pyrender 未安装，使用 trimesh 的 SceneViewer
    # trimesh SceneViewer 不支持逐帧动画更新，所以我们用 MuJoCo 来画

    # 方案: 将 SMPL mesh 逐帧写入 MuJoCo 的 user scene 作为 geom
    # 但 MuJoCo 不方便画 mesh，所以用最简单的方法:
    # 把每一帧的 SMPLX 关节 + 骨骼连线用 MuJoCo viewer 画出来

    import mujoco
    import mujoco.viewer as mjv

    # 创建一个最简的 MuJoCo 模型(只有一个地板)
    xml_str = """
    <mujoco>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="10 10 0.01" rgba="0.8 0.8 0.8 1"/>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    viewer = mjv.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    viewer.cam.distance = 3.0
    viewer.cam.elevation = -15
    viewer.cam.lookat[:] = [0, 0, 0.8]

    # 获取关节位置
    joints_all = smplx_output.joints.detach().numpy()  # (N, J, 3)
    parents = body_model.parents.numpy()

    # SMPLX 前 22 个关节是 body joints
    body_joint_count = 22

    from loop_rate_limiters import RateLimiter
    rate_limiter = RateLimiter(frequency=args.fps, warn=False)

    fps_counter = 0
    fps_start_time = time.time()

    frame_idx = 0
    while viewer.is_running():
        if frame_idx >= num_frames:
            frame_idx = 0  # 循环播放

        joints = joints_all[frame_idx]  # (J, 3)

        # 清除自定义几何体
        viewer.user_scn.ngeom = 0

        # 画关节点(球)
        for i in range(body_joint_count):
            if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                break
            geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            mujoco.mjv_initGeom(
                geom,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.02, 0.02, 0.02],
                pos=joints[i],
                mat=np.eye(3).flatten(),
                rgba=[0.2, 0.6, 1.0, 1.0],
            )
            viewer.user_scn.ngeom += 1

        # 画骨骼连线(capsule)
        for i in range(1, body_joint_count):
            parent_idx = parents[i]
            if parent_idx < 0 or parent_idx >= body_joint_count:
                continue
            if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                break

            geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            mujoco.mjv_initGeom(
                geom,
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=[0.0, 0.0, 0.0],
                pos=[0, 0, 0],
                mat=np.eye(3).flatten(),
                rgba=[0.1, 0.4, 0.8, 0.8],
            )
            mujoco.mjv_connector(
                geom,
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                width=0.015,
                from_=joints[parent_idx],
                to=joints[i],
            )
            viewer.user_scn.ngeom += 1

        # 相机跟随
        viewer.cam.lookat[:] = joints[0]  # 跟随 pelvis

        viewer.sync()
        rate_limiter.sleep()

        frame_idx += frame_skip

        # FPS 显示
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= 2.0:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"播放帧率: {actual_fps:.1f} FPS, 帧: {frame_idx}/{num_frames}")
            fps_counter = 0
            fps_start_time = current_time

    print("可视化结束")


if __name__ == "__main__":
    main()
