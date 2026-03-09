import xml.etree.ElementTree as ET

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from . import torch_utils


class Joint:
    def __init__(self, name, dof_dim, axis):
        self._name = name
        self._dof_dim = dof_dim
        self._axis = axis
        self._dof_idx = -1 # indicate the start index of dof in the whole dof vector, -1 for root or no dof joint
        
    def set_dof_idx(self, dof_idx):
        if self._dof_dim == 0:
            raise ValueError('Joint {} has no dof'.format(self._name))
        self._dof_idx = dof_idx
        
    def dof_to_rot(self, dof):
        # Input dof shape: [..., dof_dim]
        # Output rot shape: [..., 4]
        # Function: convert 1-dim or 3-dim dof to quaternion
        rot_shape = list(dof.shape[:-1]) + [4]
        ret_rot = torch.zeros(rot_shape, dtype=dof.dtype, device=dof.device)
        if self._dof_dim == 0:
            ret_rot[..., -1] = 1.0
        elif self._dof_dim == 1:
            axis = self._axis # shape: [3]
            axis = torch.broadcast_to(axis, ret_rot[..., 0:3].shape)
            ret_rot[:] = torch_utils.axis_angle_to_quat(axis, dof.squeeze(-1))
        elif self._dof_dim == 3:
            ret_rot[:] = torch_utils.exp_map_to_quat(dof)
            
        return ret_rot
    
    def rot_to_dof(self, rot):
        # Input rot shape: [..., 4]
        # Output dof shape: [..., dof_dim]
        # Function: convert quaternion to 1-dim or 3-dim dof
        dof_shape = list(rot.shape[:-1]) + [self._dof_dim]
        ret_dof = torch.zeros(dof_shape, dtype=rot.dtype, device=rot.device)
        if self._dof_dim == 1:
            axis = self._axis
            axis, angle = torch_utils.quat_to_axis_angle(rot)
            dot_axis = torch.sum(axis * self._axis, dim=-1)
            angle[dot_axis < 0] *= -1
            ret_dof[:] = angle.unsqueeze(-1)
        elif self._dof_dim == 3:
            ret_dof[:] = torch_utils.quat_to_exp_map(rot)
            
        return ret_dof
    
    @property
    def dof_dim(self):
        return self._dof_dim
    
    @property
    def name(self):
        return self._name
    
    @property
    def dof_idx(self):
        return self._dof_idx
    
    
class KinematicsModel:
    def __init__(self, file_path, device):
        self._device = device
        self._file_path = file_path
        
        self._build_kinematics_model()
        self._set_dof_indices()
        
    def _build_kinematics_model(self):
        self._body_names = []
        self._parent_indices = []
        self._local_translation = []
        self._local_rotation = []
        self._joints = []
        self._dof_size = []
        self._dof_upper_limits = []
        self._dof_lower_limits = []
        
        if self._file_path.endswith('.xml'):
            self._parse_xml()
        elif self._file_path.endswith('.urdf'):
            self._parse_urdf()
        else:
            raise NotImplementedError('File type not supported')
        
        self._parent_indices = torch.tensor(self._parent_indices, dtype=torch.long, device=self._device)
        self._local_translation = torch.tensor(np.array(self._local_translation), dtype=torch.float, device=self._device)
        self._local_rotation = torch.tensor(np.array(self._local_rotation), dtype=torch.float, device=self._device)
        self._num_dof = sum(self._dof_size)
        self._dof_lower_limits = torch.tensor(self._dof_lower_limits, dtype=torch.float, device=self._device)
        self._dof_upper_limits = torch.tensor(self._dof_upper_limits, dtype=torch.float, device=self._device)
        if self._rot_unit == "degree":
            self._dof_lower_limits = torch.deg2rad(self._dof_lower_limits)
            self._dof_upper_limits = torch.deg2rad(self._dof_upper_limits)
        
    def _parse_xml(self):
        tree = ET.parse(self._file_path)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")
        assert xml_world_body is not None, "worldbody not found"
        
        xml_body_root = xml_world_body.find("body")
        assert xml_body_root is not None, "body not found"
        
        compiler_data = xml_doc_root.find("compiler")
        self._rot_unit = compiler_data.attrib.get("angle", "degree")
        assert self._rot_unit in ["degree", "radian"], f"Invalid rotation unit: {self._rot_unit}"
        
        def _add_xml_body(xml_node, parent_index, body_index):
            body_name = xml_node.attrib.get("name")
            pos_data = xml_node.attrib.get("pos", "0 0 0")
            pos = np.fromstring(pos_data, dtype=float, sep=" ")
            
            rot_data = xml_node.attrib.get("quat", "1 0 0 0")
            rot = np.fromstring(rot_data, dtype=float, sep=" ")
            rot_w = rot[..., 0].copy()
            rot[..., 0:3] = rot[..., 1:]
            rot[..., 3] = rot_w
            
            if body_index == 0:
                curr_joint = Joint(name=body_name, dof_dim=0, axis=None) # root
            else:
                curr_joints = xml_node.findall("joint")
                num_joints = len(curr_joints)
                if num_joints == 0:
                    curr_joint = Joint(name=body_name, dof_dim=0, axis=None)
                elif num_joints == 1:
                    _axis = np.fromstring(curr_joints[0].attrib.get("axis"), dtype=float, sep=" ")
                    axis = torch.from_numpy(_axis).to(self._device)
                    curr_joint = Joint(name=body_name, dof_dim=1, axis=axis)
                    _dof_limits = np.fromstring(curr_joints[0].attrib.get("range"), dtype=float, sep=" ")
                    self._dof_lower_limits.append(_dof_limits[0])
                    self._dof_upper_limits.append(_dof_limits[1])
                elif num_joints == 3:
                    axis = None
                    curr_joint = Joint(name=body_name, dof_dim=3, axis=axis)
                    for joint in curr_joints:
                        _dof_limits = np.fromstring(joint.attrib.get("range"), dtype=float, sep=" ")
                        self._dof_lower_limits.append(_dof_limits[0])
                        self._dof_upper_limits.append(_dof_limits[1])
                else:
                    raise ValueError(f"Invalid number of joints: {num_joints} of body: {body_name}")
            
            self._body_names.append(body_name)
            self._parent_indices.append(parent_index)
            self._local_rotation.append(rot)
            self._local_translation.append(pos)
            self._joints.append(curr_joint)
            self._dof_size.append(curr_joint.dof_dim)
            
            curr_index = body_index
            body_index += 1
            for child in xml_node.findall("body"):
                body_index = _add_xml_body(child, curr_index, body_index)
                
            return body_index
        
        _add_xml_body(xml_body_root, -1, 0)

    def _parse_urdf(self):
        tree = ET.parse(self._file_path)
        urdf_doc_root = tree.getroot()
        assert urdf_doc_root.tag == "robot", "robot root not found"

        # URDF joint limits are in radians by default.
        self._rot_unit = "radian"

        links = [link.attrib["name"] for link in urdf_doc_root.findall("link")]
        children_by_parent = {}
        joint_by_child = {}
        child_links = set()

        for joint in urdf_doc_root.findall("joint"):
            joint_name = joint.attrib.get("name", "<unnamed_joint>")
            joint_type = joint.attrib.get("type", "fixed")

            parent_node = joint.find("parent")
            child_node = joint.find("child")
            assert parent_node is not None and child_node is not None, f"Invalid URDF joint: {joint_name}"
            parent_link = parent_node.attrib.get("link")
            child_link = child_node.attrib.get("link")
            assert parent_link is not None and child_link is not None, f"Invalid URDF joint link in: {joint_name}"

            origin_node = joint.find("origin")
            pos = np.zeros(3, dtype=float)
            rot = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)  # xyzw
            if origin_node is not None:
                pos = np.fromstring(origin_node.attrib.get("xyz", "0 0 0"), dtype=float, sep=" ")
                if pos.size != 3:
                    raise ValueError(f"Invalid URDF origin xyz in joint: {joint_name}")
                rpy = np.fromstring(origin_node.attrib.get("rpy", "0 0 0"), dtype=float, sep=" ")
                if rpy.size != 3:
                    raise ValueError(f"Invalid URDF origin rpy in joint: {joint_name}")
                rot = R.from_euler("xyz", rpy, degrees=False).as_quat()

            axis_node = joint.find("axis")
            axis = np.array([1.0, 0.0, 0.0], dtype=float)
            if axis_node is not None:
                axis = np.fromstring(axis_node.attrib.get("xyz", "1 0 0"), dtype=float, sep=" ")
                if axis.size != 3:
                    raise ValueError(f"Invalid URDF axis in joint: {joint_name}")
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-8:
                axis = axis / axis_norm
            else:
                axis = np.array([1.0, 0.0, 0.0], dtype=float)

            lower_limit = None
            upper_limit = None
            limit_node = joint.find("limit")
            if limit_node is not None:
                lower = limit_node.attrib.get("lower")
                upper = limit_node.attrib.get("upper")
                if lower is not None and upper is not None:
                    lower_limit = float(lower)
                    upper_limit = float(upper)

            joint_by_child[child_link] = {
                "type": joint_type,
                "parent": parent_link,
                "pos": pos,
                "rot": rot,
                "axis": axis,
                "lower": lower_limit,
                "upper": upper_limit,
            }
            child_links.add(child_link)
            children_by_parent.setdefault(parent_link, []).append(child_link)

        root_links = [link_name for link_name in links if link_name not in child_links]
        if len(root_links) != 1:
            raise ValueError(f"Expected one URDF root link, found {len(root_links)}: {root_links}")
        root_link = root_links[0]
        # Ignore a synthetic world link if the actual robot root is attached via a floating joint.
        if root_link == "world":
            world_children = children_by_parent.get(root_link, [])
            if len(world_children) == 1:
                maybe_robot_root = world_children[0]
                if joint_by_child[maybe_robot_root]["type"] == "floating":
                    root_link = maybe_robot_root

        def _add_urdf_link(link_name, parent_index, body_index):
            if parent_index == -1:
                pos = np.zeros(3, dtype=float)
                rot = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
                curr_joint = Joint(name=link_name, dof_dim=0, axis=None)
            else:
                joint_data = joint_by_child[link_name]
                pos = joint_data["pos"]
                rot = joint_data["rot"]
                joint_type = joint_data["type"]

                if joint_type in ("fixed", "floating", "planar"):
                    curr_joint = Joint(name=link_name, dof_dim=0, axis=None)
                elif joint_type in ("revolute", "continuous"):
                    axis = torch.from_numpy(joint_data["axis"]).to(self._device, dtype=torch.float)
                    curr_joint = Joint(name=link_name, dof_dim=1, axis=axis)
                    lower_limit = joint_data["lower"]
                    upper_limit = joint_data["upper"]
                    if joint_type == "continuous" or lower_limit is None or upper_limit is None:
                        lower_limit = -np.pi
                        upper_limit = np.pi
                    self._dof_lower_limits.append(lower_limit)
                    self._dof_upper_limits.append(upper_limit)
                else:
                    raise NotImplementedError(f"Unsupported URDF joint type: {joint_type}")

            self._body_names.append(link_name)
            self._parent_indices.append(parent_index)
            self._local_rotation.append(rot)
            self._local_translation.append(pos)
            self._joints.append(curr_joint)
            self._dof_size.append(curr_joint.dof_dim)

            curr_index = body_index
            body_index += 1
            for child_link in children_by_parent.get(link_name, []):
                body_index = _add_urdf_link(child_link, curr_index, body_index)

            return body_index

        _add_urdf_link(root_link, -1, 0)
        
    def _set_dof_indices(self):
        curr_dof_idx = 0
        for joint in self._joints:
            if joint.dof_dim > 0:
                joint.set_dof_idx(curr_dof_idx)
                curr_dof_idx += joint.dof_dim
                
    def dof_to_rot(self, dof):
        rot_shape = list(dof.shape[:-1]) + [self.num_joint-1, 4]
        joint_rot = torch.zeros(rot_shape, dtype=dof.dtype, device=dof.device)
        
        for j in range(1, self.num_joint):
            joint = self._joints[j]
            if joint.dof_idx == -1:
                joint_rot[..., j-1, -1] = 1.0
            else:
                joint_rot[..., j-1, :] = joint.dof_to_rot(dof[..., joint.dof_idx:joint.dof_idx+joint.dof_dim])
        return joint_rot
    
    def rot_to_dof(self, rot):
        dof_shape = list(rot.shape[:-2]) + [self.num_dof]
        dof = torch.zeros(dof_shape, dtype=rot.dtype, device=rot.device)
        
        for j in range(1, self.num_joint):
            joint = self._joints[j]
            if joint.dof_dim == 0:
                continue
            joint_rot = rot[..., j-1, :]
            dof[..., joint.dof_idx:joint.dof_idx+joint.dof_dim] = joint.rot_to_dof(joint_rot)
        
        dof = torch.clamp(dof, self._dof_lower_limits, self._dof_upper_limits)
            
        return dof
    
    def convert_local_rot_to_global(self, local_rot):
        # Input local_rot shape: [..., num_joint, 4] first row is root rotation
        # local rotation shape: [num_joint-1, 4]
        global_rot = torch.zeros_like(local_rot)
        global_rot[..., 0, :] = local_rot[..., 0, :]
        
        for j in range(1, self.num_joint):
            parent_idx = self._parent_indices[j]
            parent_rot = global_rot[..., parent_idx, :]
            local_rot_j = local_rot[..., j, :]
            global_rot[..., j, :] = torch_utils.quat_mul(parent_rot, local_rot_j)
        
        return global_rot
    
    def forward_kinematics(self, root_pos, root_rot, dof_pos, fitted_shape=None):
        joint_rot = self.dof_to_rot(dof_pos)
        
        body_pos = [None] * self.num_joint
        body_rot = [None] * self.num_joint
        
        body_pos[0] = root_pos
        body_rot[0] = root_rot
        
        for j in range(1, self.num_joint):
            j_rot = joint_rot[..., j-1, :]
            local_trans = self._local_translation[j] if fitted_shape is None else self._local_translation[j] * fitted_shape[j]
            local_rot = self._local_rotation[j]
            parent_idx = self._parent_indices[j]
            
            parent_pos = body_pos[parent_idx]
            parent_rot = body_rot[parent_idx]
            
            local_trans_broadcast = torch.broadcast_to(local_trans, parent_pos.shape)
            local_rot_broadcast = torch.broadcast_to(local_rot, parent_rot.shape)
            
            world_trans = torch_utils.quat_rotate(parent_rot, local_trans_broadcast)
            
            curr_pos = parent_pos + world_trans
            curr_rot = torch_utils.quat_mul(local_rot_broadcast, j_rot)
            curr_rot = torch_utils.quat_mul(parent_rot, curr_rot)
            
            body_pos[j] = curr_pos
            body_rot[j] = curr_rot
        
        body_pos = torch.stack(body_pos, dim=-2)
        body_rot = torch.stack(body_rot, dim=-2)
        
        return body_pos, body_rot
    
    def get_body_idx(self, body_name):
        return self._body_names.index(body_name)
    
    @property
    def body_names(self):
        return self._body_names
    
    @property
    def num_dof(self):
        return self._num_dof
    
    @property
    def num_joint(self):
        return len(self._joints)
    
    @property
    def joint_dof_idx(self):
        dof_indices = []
        for joint in self._joints:
            dof_indices.append(joint.dof_idx)
        return dof_indices
    
    @property
    def parent_indices(self):
        return self._parent_indices
    
    def get_parent_idx(self, idx):
        return self._parent_indices[idx]
    
    def get_dof_limits(self):
        return self._dof_lower_limits, self._dof_upper_limits
