import argparse
import pathlib
import os
import numpy as np
from tqdm import tqdm
import torch
import pickle

from general_motion_retargeting.utils.lafan1 import load_lafan1_file
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.motion_export import build_npz_motion_data
from rich import print


if __name__ == "__main__":
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_folder",
        help="Folder containing BVH motion files to load.",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--tgt_folder",
        help="Folder to save the retargeted motion files.",
        default="../../motion_data/LAFAN1_g1_gmr"
    )
    
    parser.add_argument(
        "--robot",
        default="unitree_g1",
    )

    parser.add_argument(
        "--output_format",
        choices=["pkl", "npz", "both"],
        default="npz",
        help="Output format for converted motions.",
    )
    
    parser.add_argument(
        "--override",
        default=False,
        action="store_true",
    )
    
    parser.add_argument(
        "--target_fps",
        default=30,
        type=int,
    )

    args = parser.parse_args()
    
    src_folder = args.src_folder
    tgt_folder = args.tgt_folder
    fk_device = "cuda:0" if torch.cuda.is_available() else "cpu"

   
   
        
    # walk over all files in src_folder
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in tqdm(sorted(filenames), desc="Retargeting files"):
            if not filename.endswith(".bvh"):
                continue
                
            # get the bvh file path
            bvh_file_path = os.path.join(dirpath, filename)
            
            tgt_base_path = bvh_file_path.replace(src_folder, tgt_folder).replace(".bvh", "")
            tgt_pkl_path = tgt_base_path + ".pkl"
            tgt_npz_path = tgt_base_path + ".npz"

            needs_pkl = args.output_format in ["pkl", "both"]
            needs_npz = args.output_format in ["npz", "both"]
            pkl_exists = os.path.exists(tgt_pkl_path)
            npz_exists = os.path.exists(tgt_npz_path)

            if not args.override:
                if needs_pkl and needs_npz and pkl_exists and npz_exists:
                    print(f"Skipping {bvh_file_path} because both outputs already exist")
                    continue
                if needs_pkl and not needs_npz and pkl_exists:
                    print(f"Skipping {bvh_file_path} because {tgt_pkl_path} exists")
                    continue
                if needs_npz and not needs_pkl and npz_exists:
                    print(f"Skipping {bvh_file_path} because {tgt_npz_path} exists")
                    continue
            
            # Load LAFAN1 trajectory
            try:
                lafan1_data_frames, actual_human_height = load_lafan1_file(bvh_file_path)
                src_fps = 30  # LAFAN1 data is typically 30 FPS
            except Exception as e:
                print(f"Error loading {bvh_file_path}: {e}")
                continue

            
            # Initialize the retargeting system
            retarget = GMR(
                src_human="bvh_lafan1",
                tgt_robot=args.robot,
                actual_human_height=actual_human_height,
            )

            # retarget to get all qpos
            qpos_list = []
            for curr_frame in range(len(lafan1_data_frames)):
                smplx_data = lafan1_data_frames[curr_frame]
                
                # Retarget till convergence
                qpos = retarget.retarget(smplx_data)
                
                qpos_list.append(qpos.copy())
            
            qpos_list = np.array(qpos_list)

            if needs_npz:
                npz_data = build_npz_motion_data(
                    retarget.xml_file,
                    qpos_list,
                    src_fps,
                    fk_device=fk_device,
                )
                os.makedirs(os.path.dirname(tgt_npz_path), exist_ok=True)
                np.savez(tgt_npz_path, **npz_data)

            if needs_pkl:
                kinematics_model = KinematicsModel(retarget.xml_file, device=fk_device)

                root_pos = qpos_list[:, :3]
                root_rot = qpos_list[:, 3:7]
                root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]
                dof_pos = qpos_list[:, 7:]
                num_frames = root_pos.shape[0]

                identity_root_pos = torch.zeros((num_frames, 3), device=fk_device)
                identity_root_rot = torch.zeros((num_frames, 4), device=fk_device)
                identity_root_rot[:, -1] = 1.0
                local_body_pos, _ = kinematics_model.forward_kinematics(
                    identity_root_pos,
                    identity_root_rot,
                    torch.from_numpy(dof_pos).to(device=fk_device, dtype=torch.float),
                )
                body_names = kinematics_model.body_names

                motion_data = {
                    "root_pos": root_pos,
                    "root_rot": root_rot,
                    "dof_pos": dof_pos,
                    "local_body_pos": local_body_pos.detach().cpu().numpy(),
                    "fps": src_fps,
                    "link_body_list": body_names,
                }

                os.makedirs(os.path.dirname(tgt_pkl_path), exist_ok=True)
                with open(tgt_pkl_path, "wb") as f:
                    pickle.dump(motion_data, f)

    print("Done. saved to ", tgt_folder)
