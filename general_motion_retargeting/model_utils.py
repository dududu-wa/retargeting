from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET

from scipy.spatial.transform import Rotation as R

from .params import ASSET_ROOT, ROBOT_XML_DICT


UNITREE_G1_BVH_URDF_PATH = (
    ASSET_ROOT / "unitree_g1" / "g1_custom_collision_29dof.urdf"
)


def resolve_robot_model_path(
    tgt_robot: str,
    src_human: str | None = None,
    robot_model_path: str | pathlib.Path | None = None,
) -> pathlib.Path:
    if robot_model_path is None and _should_use_unitree_g1_bvh_model(
        tgt_robot=tgt_robot,
        src_human=src_human,
    ):
        robot_model_path = UNITREE_G1_BVH_URDF_PATH

    if robot_model_path is None:
        return pathlib.Path(ROBOT_XML_DICT[tgt_robot]).resolve()

    model_path = pathlib.Path(robot_model_path).expanduser().resolve()
    if model_path.suffix.lower() == ".urdf" and tgt_robot == "unitree_g1":
        return generate_unitree_g1_mjcf_from_urdf(model_path)

    return model_path


def _should_use_unitree_g1_bvh_model(tgt_robot: str, src_human: str | None) -> bool:
    return (
        tgt_robot == "unitree_g1"
        and isinstance(src_human, str)
        and src_human.startswith("bvh")
        and UNITREE_G1_BVH_URDF_PATH.exists()
    )


def generate_unitree_g1_mjcf_from_urdf(urdf_path: str | pathlib.Path) -> pathlib.Path:
    urdf_path = pathlib.Path(urdf_path).expanduser().resolve()
    base_xml_path = pathlib.Path(ROBOT_XML_DICT["unitree_g1"]).resolve()
    generated_xml_path = urdf_path.with_suffix(".generated.xml")

    source_mtime = max(base_xml_path.stat().st_mtime, urdf_path.stat().st_mtime)
    if generated_xml_path.exists() and generated_xml_path.stat().st_mtime >= source_mtime:
        return generated_xml_path

    base_tree = ET.parse(base_xml_path)
    base_root = base_tree.getroot()
    body_nodes = {
        body.attrib["name"]: body
        for body in base_root.findall(".//body[@name]")
    }
    collisions_by_link = _parse_urdf_collisions(urdf_path)

    for body_name, collision_geoms in collisions_by_link.items():
        body_node = body_nodes.get(body_name)
        if body_node is None:
            continue

        _remove_default_collision_geoms(body_node)
        for geom in collision_geoms:
            body_node.append(geom)

    if hasattr(ET, "indent"):
        ET.indent(base_tree, space="  ")
    base_tree.write(generated_xml_path, encoding="utf-8")
    return generated_xml_path


def _parse_urdf_collisions(urdf_path: pathlib.Path) -> dict[str, list[ET.Element]]:
    urdf_tree = ET.parse(urdf_path)
    urdf_root = urdf_tree.getroot()
    collisions_by_link: dict[str, list[ET.Element]] = {}

    for link_node in urdf_root.findall("link"):
        link_name = link_node.attrib.get("name")
        if not link_name:
            continue

        link_geoms = []
        for collision_index, collision_node in enumerate(link_node.findall("collision")):
            geom_node = _urdf_collision_to_mjcf_geom(
                link_name=link_name,
                collision_index=collision_index,
                collision_node=collision_node,
            )
            if geom_node is not None:
                link_geoms.append(geom_node)

        if link_geoms:
            collisions_by_link[link_name] = link_geoms

    return collisions_by_link


def _urdf_collision_to_mjcf_geom(
    link_name: str,
    collision_index: int,
    collision_node: ET.Element,
) -> ET.Element | None:
    geometry_node = collision_node.find("geometry")
    if geometry_node is None:
        return None

    geom_node = ET.Element("geom")
    collision_name = collision_node.attrib.get("name")
    if collision_name:
        geom_node.set("name", collision_name)
    else:
        geom_node.set("name", f"{link_name}_collision_{collision_index}")

    origin_node = collision_node.find("origin")
    if origin_node is not None:
        xyz = origin_node.attrib.get("xyz")
        if xyz:
            geom_node.set("pos", xyz)

        rpy = origin_node.attrib.get("rpy")
        if rpy:
            quat = _rpy_to_mujoco_quat(rpy)
            geom_node.set("quat", quat)

    sphere_node = geometry_node.find("sphere")
    if sphere_node is not None:
        geom_node.set("type", "sphere")
        geom_node.set("size", sphere_node.attrib["radius"])
        return geom_node

    cylinder_node = geometry_node.find("cylinder")
    if cylinder_node is not None:
        radius = float(cylinder_node.attrib["radius"])
        half_length = float(cylinder_node.attrib["length"]) / 2.0
        geom_node.set("type", "cylinder")
        geom_node.set("size", f"{radius:.12g} {half_length:.12g}")
        return geom_node

    box_node = geometry_node.find("box")
    if box_node is not None:
        half_extents = [
            float(value) / 2.0 for value in box_node.attrib["size"].split()
        ]
        geom_node.set("type", "box")
        geom_node.set(
            "size",
            " ".join(f"{value:.12g}" for value in half_extents),
        )
        return geom_node

    mesh_node = geometry_node.find("mesh")
    if mesh_node is not None:
        geom_node.set("type", "mesh")
        geom_node.set("mesh", link_name)
        return geom_node

    return None


def _remove_default_collision_geoms(body_node: ET.Element) -> None:
    for geom_node in list(body_node.findall("geom")):
        if _is_visual_geom(geom_node):
            continue
        body_node.remove(geom_node)


def _is_visual_geom(geom_node: ET.Element) -> bool:
    return (
        geom_node.attrib.get("group") == "1"
        or geom_node.attrib.get("contype") == "0"
        or geom_node.attrib.get("conaffinity") == "0"
        or geom_node.attrib.get("density") == "0"
    )


def _rpy_to_mujoco_quat(rpy: str) -> str:
    roll, pitch, yaw = (float(value) for value in rpy.split())
    quat_xyzw = R.from_euler("xyz", [roll, pitch, yaw]).as_quat()
    quat_wxyz = [
        quat_xyzw[3],
        quat_xyzw[0],
        quat_xyzw[1],
        quat_xyzw[2],
    ]
    return " ".join(f"{value:.12g}" for value in quat_wxyz)
