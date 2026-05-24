
import mink
import mujoco as mj
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT
from rich import print

class BodyDirectionTask(mink.Task):
    """Align the direction from one robot body to another with a target vector."""

    def __init__(
        self,
        model: mj.MjModel,
        start_body_name: str,
        end_body_name: str,
        cost,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        self.model = model
        self.start_body_name = start_body_name
        self.end_body_name = end_body_name
        self.start_body_id = mj.mj_name2id(
            model, mj.mjtObj.mjOBJ_BODY, start_body_name
        )
        self.end_body_id = mj.mj_name2id(
            model, mj.mjtObj.mjOBJ_BODY, end_body_name
        )
        if self.start_body_id < 0:
            raise ValueError(f"Unknown direction-task start body: {start_body_name}")
        if self.end_body_id < 0:
            raise ValueError(f"Unknown direction-task end body: {end_body_name}")

        cost = np.broadcast_to(np.atleast_1d(cost).astype(float), (3,)).copy()
        super().__init__(cost=cost, gain=gain, lm_damping=lm_damping)
        self.target_direction = None

    def set_target_direction(self, direction):
        direction = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            raise ValueError(
                f"Direction target for {self.start_body_name}->{self.end_body_name} is near zero"
            )
        self.target_direction = direction / norm

    def _current_direction(self, configuration):
        start = configuration.data.xpos[self.start_body_id]
        end = configuration.data.xpos[self.end_body_id]
        delta = end - start
        norm = max(np.linalg.norm(delta), 1e-8)
        return delta / norm, norm

    def compute_error(self, configuration):
        if self.target_direction is None:
            raise ValueError(
                f"Direction target is not set for {self.start_body_name}->{self.end_body_name}"
            )
        current, _ = self._current_direction(configuration)
        return current - self.target_direction

    def compute_jacobian(self, configuration):
        current, length = self._current_direction(configuration)
        start_jac = np.zeros((3, configuration.model.nv))
        end_jac = np.zeros((3, configuration.model.nv))
        unused_rot_jac = np.zeros((3, configuration.model.nv))
        mj.mj_jacBody(
            configuration.model,
            configuration.data,
            start_jac,
            unused_rot_jac,
            self.start_body_id,
        )
        mj.mj_jacBody(
            configuration.model,
            configuration.data,
            end_jac,
            unused_rot_jac,
            self.end_body_id,
        )

        # For u = v / ||v||, du/dq = (I - uu^T) / ||v|| * dv/dq.
        # This is the standard differential-kinematics form used in
        # resolved-rate IK (e.g. Siciliano et al., Robotics, 2009) and avoids
        # constraining the no-hand R2V2 wrist's full orientation.
        projector = np.eye(3) - np.outer(current, current)
        return (projector / length) @ (end_jac - start_jac)


class GeneralMotionRetargeting:
    """General Motion Retargeting (GMR).
    """
    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float = None,
        solver: str="daqp", # change from "quadprog" to "daqp".
        damping: float=5e-1, # change from 1e-1 to 1e-2.
        verbose: bool=True,
        use_velocity_limit: bool=False,
    ) -> None:

        # load the robot model
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)
        
        # Print DoF names in order
        print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
        self.robot_dof_names = {}
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            if verbose:
                print(f"DoF {i}: {dof_name}")
            
            
        print("[GMR] Robot Body names and their IDs:")
        self.robot_body_names = {}
        for i in range(self.model.nbody):  # 'nbody' is the number of bodies
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            if verbose:
                print(f"Body ID {i}: {body_name}")
        
        print("[GMR] Robot Motor (Actuator) names and their IDs:")
        self.robot_motor_names = {}
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            if verbose:
                print(f"Motor ID {i}: {motor_name}")

        # Load the IK config
        with open(IK_CONFIG_DICT[src_human][tgt_robot]) as f:
            ik_config = json.load(f)
        if verbose:
            print("Use IK config: ", IK_CONFIG_DICT[src_human][tgt_robot])
        
        # compute the scale ratio based on given human height and the assumption in the IK config
        if actual_human_height is not None:
            ratio = actual_human_height / ik_config["human_height_assumption"]
        else:
            ratio = 1.0
            
        # adjust the human scale table
        for key in ik_config["human_scale_table"].keys():
            ik_config["human_scale_table"][key] = ik_config["human_scale_table"][key] * ratio
    

        # used for retargeting
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        self.human_scale_table = ik_config["human_scale_table"]
        self.posture_task_config = ik_config.get("posture_task", {"enabled": False})
        self.direction_task_config = ik_config.get("direction_tasks", [])
        self.adaptive_task_cost_config = ik_config.get("adaptive_task_costs", {"enabled": False})
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])

        self.max_iter = 10

        self.solver = solver
        self.damping = damping

        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}

        self.task_errors1 = {}
        self.task_errors2 = {}
        self.human_bodies_to_direction_tasks = []
        self.frame_tasks2_by_robot_body = {}
        self.frame_task2_costs = {}

        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        if use_velocity_limit:
            VELOCITY_LIMITS = {k: 3*np.pi for k in self.robot_motor_names.keys()}
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS)) 
            
        self.setup_retarget_configuration()
        
        self.ground_offset = 0.0

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)
    
        self.tasks1 = []
        self.tasks2 = []
        
        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task1[body_name] = task
                self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets1[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks1.append(task)
                self.task_errors1[task] = []
        
        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task2[body_name] = task
                self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets2[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks2.append(task)
                self.task_errors2[task] = []
                self.frame_tasks2_by_robot_body[frame_name] = task
                self.frame_task2_costs[frame_name] = (pos_weight, rot_weight)

        posture_task = self.create_posture_task()
        if posture_task is not None:
            self.tasks1.append(posture_task)
            self.tasks2.append(posture_task)

        for entry in self.direction_task_config:
            task = BodyDirectionTask(
                model=self.model,
                start_body_name=entry["robot_start_body"],
                end_body_name=entry["robot_end_body"],
                cost=entry.get("cost", 1.0),
                gain=float(entry.get("gain", 1.0)),
                lm_damping=float(entry.get("lm_damping", 0.0)),
            )
            # Add the direction task only in the second pass so torso and
            # shoulder placement are established before forearm direction is
            # refined. This follows the project two-stage IK structure.
            self.tasks2.append(task)
            self.human_bodies_to_direction_tasks.append(
                (entry["human_start_body"], entry["human_end_body"], task)
            )

    def create_posture_task(self):
        if not self.posture_task_config.get("enabled", False):
            return None

        cost = np.zeros(self.model.nv)
        # The mink PostureTask is a low-priority neutral-pose regularizer; this
        # keeps hand/forearm tracking from borrowing waist motion in redundant
        # or conflicting IK solves. Reference: https://kevinzakka.github.io/mink/
        for joint_name, joint_cost in self.posture_task_config.get("costs", {}).items():
            joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"posture_task references unknown joint: {joint_name}")
            if int(self.model.jnt_type[joint_id]) == mj.mjtJoint.mjJNT_FREE:
                continue
            dof_id = int(self.model.jnt_dofadr[joint_id])
            cost[dof_id] = float(joint_cost)

        task = mink.PostureTask(
            self.model,
            cost=cost,
            gain=float(self.posture_task_config.get("gain", 1.0)),
            lm_damping=float(self.posture_task_config.get("lm_damping", 0.0)),
        )

        target = self.posture_task_config.get("target", "qpos0")
        if target != "qpos0":
            raise ValueError(f"Unsupported posture_task target: {target}")

        target_qpos = self.model.qpos0.copy()
        # PostureTask targets live in MuJoCo qpos space. Small configured
        # offsets let us bias redundant arm joints without changing torso
        # FrameTask weights, following mink's low-priority posture regularizer.
        for joint_name, qpos_offset in self.posture_task_config.get("target_offsets", {}).items():
            joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"posture_task target_offsets references unknown joint: {joint_name}")
            if int(self.model.jnt_type[joint_id]) == mj.mjtJoint.mjJNT_FREE:
                raise ValueError(f"posture_task target_offsets cannot target free joint: {joint_name}")
            qpos_id = int(self.model.jnt_qposadr[joint_id])
            target_qpos[qpos_id] += float(qpos_offset)

        task.set_target(target_qpos)
        return task

  
    def update_targets(self, human_data, offset_to_ground=False):
        # scale human data in local frame
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(human_data, self.human_root_name, self.human_scale_table)
        human_data = self.offset_human_data(human_data, self.pos_offsets1, self.rot_offsets1)
        human_data = self.apply_ground_offset(human_data)
        if offset_to_ground:
            human_data = self.offset_human_data_to_ground(human_data)
        self.scaled_human_data = human_data

        if self.use_ik_match_table1:
            for body_name in self.human_body_to_task1.keys():
                task = self.human_body_to_task1[body_name]
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
        
        if self.use_ik_match_table2:
            for body_name in self.human_body_to_task2.keys():
                task = self.human_body_to_task2[body_name]
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))

        for human_start, human_end, task in self.human_bodies_to_direction_tasks:
            start_pos = human_data[human_start][0]
            end_pos = human_data[human_end][0]
            task.set_target_direction(end_pos - start_pos)

        self.update_adaptive_task_costs(human_data)

    def update_adaptive_task_costs(self, human_data):
        if not self.adaptive_task_cost_config.get("enabled", False):
            return

        foot_names = self.adaptive_task_cost_config.get(
            "human_foot_bodies", ["LeftFootMod", "RightFootMod"]
        )
        head_name = self.adaptive_task_cost_config.get("human_head_body", "Spine2")
        if self.human_root_name not in human_data or head_name not in human_data:
            return
        if any(name not in human_data for name in foot_names):
            return

        foot_z = min(float(human_data[name][0][2]) for name in foot_names)
        hip_rel = float(human_data[self.human_root_name][0][2] - foot_z)
        head_rel = float(human_data[head_name][0][2] - foot_z)
        low_pose = (
            hip_rel < float(self.adaptive_task_cost_config.get("hip_height_threshold", 0.35))
            or head_rel < float(self.adaptive_task_cost_config.get("head_height_threshold", 0.85))
        )

        for frame_name, scales in self.adaptive_task_cost_config.get("frame_scales", {}).items():
            task = self.frame_tasks2_by_robot_body.get(frame_name)
            if task is None:
                continue
            base_position_cost, base_orientation_cost = self.frame_task2_costs[frame_name]
            if low_pose:
                position_cost = base_position_cost * float(scales.get("position", 1.0))
                orientation_cost = base_orientation_cost * float(scales.get("orientation", 1.0))
            else:
                position_cost = base_position_cost
                orientation_cost = base_orientation_cost
            # Mink FrameTask exposes cost setters, so low-pose retargeting can
            # reduce over-constraining contacts while retaining normal gait
            # foot tracking outside floor/roll poses.
            task.set_position_cost(position_cost)
            task.set_orientation_cost(orientation_cost)
            
            
    def retarget(self, human_data, offset_to_ground=False):
        # Update the task targets
        self.update_targets(human_data, offset_to_ground)

        if self.use_ik_match_table1:
            # Solve the IK problem
            curr_error = self.error1()
            dt = self.configuration.model.opt.timestep
            vel1 = mink.solve_ik(
                self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel1, dt)
            next_error = self.error1()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                dt = self.configuration.model.opt.timestep
                vel1 = mink.solve_ik(
                    self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel1, dt)
                next_error = self.error1()
                num_iter += 1

        if self.use_ik_match_table2:
            curr_error = self.error2()
            dt = self.configuration.model.opt.timestep
            vel2 = mink.solve_ik(
                self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel2, dt)
            next_error = self.error2()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                # Solve the IK problem with the second task
                dt = self.configuration.model.opt.timestep
                vel2 = mink.solve_ik(
                    self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel2, dt)
                
                next_error = self.error2()
                num_iter += 1
                
            
        return self.configuration.data.qpos.copy()


    def error1(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks1]
            )
        )
    
    def error2(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks2]
            )
        )


    def to_numpy(self, human_data):
        for body_name in human_data.keys():
            human_data[body_name] = [np.asarray(human_data[body_name][0]), np.asarray(human_data[body_name][1])]
        return human_data


    def scale_human_data(self, human_data, human_root_name, human_scale_table):
        
        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]
        
        # scale root
        scaled_root_pos = human_scale_table[human_root_name] * root_pos
        
        # scale other body parts in local frame
        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                continue
            else:
                # transform to local frame (only position)
                human_data_local[body_name] = (human_data[body_name][0] - root_pos) * human_scale_table[body_name]
            
        # transform the human data back to the global frame
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (human_data_local[body_name] + scaled_root_pos, human_data[body_name][1])

        return human_data_global
    
    def offset_human_data(self, human_data, pos_offsets, rot_offsets):
        """the pos offsets are applied in the local frame"""
        offset_human_data = {}
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            # Zero-cost IK entries do not create FrameTask offsets; keep their
            # scaled pose available without trying to apply a missing offset.
            if body_name not in pos_offsets or body_name not in rot_offsets:
                continue
            # apply rotation offset first
            updated_quat = (R.from_quat(quat, scalar_first=True) * rot_offsets[body_name]).as_quat(scalar_first=True)
            offset_human_data[body_name][1] = updated_quat
            
            local_offset = pos_offsets[body_name]
            # compute the global position offset using the updated rotation
            global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(local_offset)
            
            offset_human_data[body_name][0] = pos + global_pos_offset
           
        return offset_human_data
            
    def offset_human_data_to_ground(self, human_data):
        """find the lowest point of the human data and offset the human data to the ground"""
        offset_human_data = {}
        ground_offset = 0.1
        lowest_pos = np.inf

        for body_name in human_data.keys():
            # only consider the foot/Foot
            if "Foot" not in body_name and "foot" not in body_name:
                continue
            pos, quat = human_data[body_name]
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
                lowest_body_name = body_name
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            offset_human_data[body_name][0] = pos - np.array([0, 0, lowest_pos]) + np.array([0, 0, ground_offset])
        return offset_human_data

    def set_ground_offset(self, ground_offset):
        self.ground_offset = ground_offset

    def apply_ground_offset(self, human_data):
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            human_data[body_name][0] = pos - np.array([0, 0, self.ground_offset])
        return human_data
