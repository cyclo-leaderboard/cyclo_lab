"""Task000901 real-aligned WhiteRoom scene with the SG2 reference pose."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import torch
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import from_files
from isaaclab.sim.utils import bind_physics_material, clone, find_matching_prim_paths
from isaaclab.sim.spawners.sensors import spawn_camera
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.events import reset_scene_to_default
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from cyclo_lab.assets.robots.FFW_SG2 import (
    FFW_SG2_CFG,
    spawn_sg2_with_softgripper_physics,
)
from _cyclo_task000901_test4_contact import PROFILE as TEST4_PROFILE
from _cyclo_task000901_test4_contact import apply_robot_profile as apply_test4_robot_profile
from cyclo_lab.assets.sensors.ffw_sg2_cameras import (
    make_ffw_sg2_head_camera_cfg,
    make_ffw_sg2_wrist_camera_cfg,
)
_ROOT=Path(__file__).resolve().parent
_MANUAL_SPEC = importlib.util.spec_from_file_location(
    "_cyclo_task000901_manual_profile", _ROOT / "manual_calibration/task901_profile.py"
)
_MANUAL_MODULE = importlib.util.module_from_spec(_MANUAL_SPEC)
_MANUAL_SPEC.loader.exec_module(_MANUAL_MODULE)
_TASK901_MANUAL_PROFILE = (
    None if os.environ.get("TASK901_CALIBRATION_PROFILE_MODE") == "skip"
    else _MANUAL_MODULE.load()
)
_PROFILE_SPEC = importlib.util.spec_from_file_location(
    "_cyclo_task000901_camera_color_profile", _ROOT / "camera_color_profile.py"
)
_PROFILE_MODULE = importlib.util.module_from_spec(_PROFILE_SPEC)
_PROFILE_SPEC.loader.exec_module(_PROFILE_MODULE)
apply_camera_profile = _PROFILE_MODULE.apply_camera_profile
if _TASK901_MANUAL_PROFILE is not None:
    _MANUAL_MODULE.apply_color(_PROFILE_MODULE.PROFILES, _TASK901_MANUAL_PROFILE)
TABLE_TOP_Z_M=.688
PLATE_HOLDER_TABLE_TOP_Z_M=TABLE_TOP_Z_M+TEST4_PROFILE["table_height_offset_m"]
ROOM_USD_PATH=str(_ROOT/'runtime/data/white_room_task000901_visual_v2.usda')
_MANUAL_LIGHTING_PATH = _ROOT / "manual_calibration/applied_lighting.usda"
LIGHTING_USD_PATH = str(
    _MANUAL_LIGHTING_PATH if _TASK901_MANUAL_PROFILE is not None and _MANUAL_LIGHTING_PATH.exists()
    else _ROOT / 'runtime/data/lighting_task000901_visual_v2.usda'
)
PRODUCE_USD_DIR=_ROOT/'runtime/data/objects/visual_v2/produce'
COOKWARE_USD_DIR=_ROOT/'runtime/data/objects/visual_v2/cookware'
PLATE_USD_DIR=_ROOT/'runtime/data/objects/plates_v1'
PLATE_HOLDER_USD_PATH=_ROOT/'runtime/data/objects/plate_holder_v1/wood_plate_holder.usda'
PLATE_HOLDER_ROOM_USD_PATH=_ROOT/'runtime/data/white_room_plate_holder_table685.usda'
PLATE_HOLDER_LIGHTING_USD_PATH=_ROOT/'runtime/data/white_room_plate_holder_lighting.usda'

# Live 1050 SG2 state recaptured 2026-11-02. Wheel encoder positions are not
# copied because they do not affect the static alignment view.
SG2_REFERENCE_JOINT_POS = {
    "arm_l_joint1": 1.0,
    "arm_l_joint2": 0.0,
    "arm_l_joint3": 0.0,
    "arm_l_joint4": -2.2,
    "arm_l_joint5": 0.0,
    "arm_l_joint6": 0.0,
    "arm_l_joint7": 0.0,
    "arm_r_joint1": 1.0,
    "arm_r_joint2": 0.0,
    "arm_r_joint3": 0.0,
    "arm_r_joint4": -2.2,
    "arm_r_joint5": 0.0,
    "arm_r_joint6": 0.0,
    "arm_r_joint7": 0.0,
    "gripper_l_joint1": 0.0,
    "gripper_r_joint1": 0.0,
    "head_joint1": 0.8299,
    "head_joint2": 0.0,
    "lift_joint": -0.2,
}

# Keep the recorded lift at -0.2 m and the robot base on the floor. The head
# camera and tabletop height are fixed while fitting table edges and holder rails.
SG2_ROOT_POS = (-0.0274283483, -0.1173307863, 0.0)
SG2_ROOT_ROT = (-0.0005745888, 0.0, 0.0, 0.9999998349)


class Task000901HeadCamera(Camera):
    """Apply the calibrated live head-camera color response."""

    def _update_intrinsic_matrices(self, env_ids):
        super()._update_intrinsic_matrices(env_ids)
        for index in env_ids:
            intrinsic = self._data.intrinsic_matrices[index]
            intrinsic[0, 0] = TASK000901_SIM_HEAD_K[0]
            intrinsic[0, 2] = TASK000901_SIM_HEAD_K[2]
            intrinsic[1, 1] = TASK000901_SIM_HEAD_K[4]
            intrinsic[1, 2] = TASK000901_SIM_HEAD_K[5]

    def _update_buffers_impl(self, env_ids):
        super()._update_buffers_impl(env_ids)
        rgb = self._data.output.get("rgb")
        if rgb is None:
            return
        if os.environ.get("TASK901_CAMERA_PROFILE_MODE") == "raw":
            return
        corrected = apply_camera_profile(rgb[env_ids], "head")
        rgb[env_ids] = corrected.clamp_(0, 255).to(rgb.dtype)


TASK000901_ZED_HEAD_K = (
    734.6712036132812, 0.0, 626.33935546875,
    0.0, 734.6712036132812, 360.0394592285156,
    0.0, 0.0, 1.0,
)
# Refined against the tabletop outlines in MCAP episodes 20, 21 and 22.
# Preserve the previously fitted principal point while changing focal length.
TASK000901_HEAD_ARM_FOCAL_SCALE = 1.02340385912016
TASK000901_SIM_HEAD_CY = 363.04144287109375
TASK000901_SIM_HEAD_K = (
    TASK000901_ZED_HEAD_K[0] * TASK000901_HEAD_ARM_FOCAL_SCALE,
    0.0, TASK000901_ZED_HEAD_K[2],
    0.0, TASK000901_ZED_HEAD_K[4] * TASK000901_HEAD_ARM_FOCAL_SCALE,
    TASK000901_SIM_HEAD_CY,
    0.0, 0.0, 1.0,
)

# Live 1050 D405 color camera_info captured on 2026-11-12. The recorder uses
# image_rect_raw at 640x480 and then rotates the saved wrist frame 270 degrees
# to 480x640: x' = y, y' = 639 - x. Keep the native D values as provenance,
# but do not distort the already-rectified recording target a second time.
TASK000901_D405_NATIVE_K = {
    "left": (
        392.85992431640625, 0.0, 321.6305847167969,
        0.0, 392.2402038574219, 242.0066680908203,
        0.0, 0.0, 1.0,
    ),
    "right": (
        392.4100646972656, 0.0, 315.6520080566406,
        0.0, 391.906982421875, 240.0281982421875,
        0.0, 0.0, 1.0,
    ),
}
TASK000901_D405_NATIVE_D = {
    "left": (-0.052548039704561234, 0.0581633560359478, 0.00010541319352341816,
             0.000825238530524075, -0.018877947703003883),
    "right": (-0.055024340748786926, 0.0616723895072937, -0.00016217156371567398,
              0.0004359635931905359, -0.02069825306534767),
}


def _rotate_wrist_k_270(native_k):
    return (
        native_k[4], 0.0, native_k[5],
        0.0, native_k[0], 639.0 - native_k[2],
        0.0, 0.0, 1.0,
    )


TASK000901_SIM_WRIST_K = {
    side: _rotate_wrist_k_270(native_k)
    for side, native_k in TASK000901_D405_NATIVE_K.items()
}


def spawn_task000901_zed_head_camera(prim_path, cfg, *args, **kwargs):
    """Spawn head views with arm-fitted focal scale and lower-image anchor."""
    spawned = spawn_camera(prim_path, cfg, *args, **kwargs)
    for path in find_matching_prim_paths(prim_path, spawned.GetStage()):
        prim = spawned.GetStage().GetPrimAtPath(path)
        prim.ApplyAPI("OmniLensDistortionOpenCvPinholeAPI")
        prim.GetAttribute("omni:lensdistortion:model").Set("opencvPinhole")
        for key, value in {
            "fx": TASK000901_SIM_HEAD_K[0],
            "fy": TASK000901_SIM_HEAD_K[4],
            "cx": TASK000901_SIM_HEAD_K[2],
            "cy": TASK000901_SIM_HEAD_K[5],
            "imageSize": Gf.Vec2i(1280, 720),
        }.items():
            prim.GetAttribute(f"omni:lensdistortion:opencvPinhole:{key}").Set(value)
        for key in ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2", "s1", "s2", "s3", "s4"):
            prim.GetAttribute(f"omni:lensdistortion:opencvPinhole:{key}").Set(0.0)
    return spawned


class Task000901WristCamera(Camera):
    """Apply the separately fitted left/right live wrist-camera response."""

    def _update_intrinsic_matrices(self, env_ids):
        super()._update_intrinsic_matrices(env_ids)
        side = "left" if "wrist_left" in self.cfg.prim_path else "right"
        target = TASK000901_SIM_WRIST_K[side]
        for index in env_ids:
            intrinsic = self._data.intrinsic_matrices[index]
            intrinsic[0, 0] = target[0]
            intrinsic[0, 2] = target[2]
            intrinsic[1, 1] = target[4]
            intrinsic[1, 2] = target[5]

    def _update_buffers_impl(self, env_ids):
        super()._update_buffers_impl(env_ids)
        rgb = self._data.output.get("rgb")
        if rgb is None:
            return
        if os.environ.get("TASK901_CAMERA_PROFILE_MODE") == "raw":
            return
        camera = "wrist_left" if "wrist_left" in self.cfg.prim_path else "wrist_right"
        corrected = apply_camera_profile(rgb[env_ids], camera)
        rgb[env_ids] = corrected.clamp_(0, 255).to(rgb.dtype)


def spawn_task000901_d405_wrist_camera(prim_path, cfg, *args, **kwargs):
    """Render the 270-degree-rotated rectified 1050 D405 intrinsic exactly."""
    spawned = spawn_camera(prim_path, cfg, *args, **kwargs)
    side = "left" if "wrist_left" in prim_path else "right"
    target = TASK000901_SIM_WRIST_K[side]
    for path in find_matching_prim_paths(prim_path, spawned.GetStage()):
        prim = spawned.GetStage().GetPrimAtPath(path)
        prim.ApplyAPI("OmniLensDistortionOpenCvPinholeAPI")
        prim.GetAttribute("omni:lensdistortion:model").Set("opencvPinhole")
        for key, value in {
            "fx": target[0],
            "fy": target[4],
            "cx": target[2],
            "cy": target[5],
            "imageSize": Gf.Vec2i(480, 640),
        }.items():
            prim.GetAttribute(f"omni:lensdistortion:opencvPinhole:{key}").Set(value)
        for key in ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2", "s1", "s2", "s3", "s4"):
            prim.GetAttribute(f"omni:lensdistortion:opencvPinhole:{key}").Set(0.0)
    return spawned


def spawn_task000901_test4_robot(prim_path, cfg, *args, **kwargs):
    """Spawn the shared SG2, then apply the Task901 real-to-sim contact profile."""
    result = spawn_sg2_with_softgripper_physics(prim_path, cfg, *args, **kwargs)
    apply_test4_robot_profile(result.GetStage(), str(result.GetPath()))
    # Keep the existing 0.5/0.5 proximal-link coefficients, but use max
    # against plates and the holder, just like the distal tip material.
    stage = result.GetStage()
    link_material_path = f"{result.GetPath()}/ProximalGripperPhysicsMaterial"
    link_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.5, dynamic_friction=0.5, restitution=0.0,
        friction_combine_mode=("average" if os.environ.get("TASK901_PHYSICS_PROFILE") == "test4" else "max"),
        restitution_combine_mode="min",
    )
    link_material.func(link_material_path, link_material)
    for collision in Usd.PrimRange(result):
        path = str(collision.GetPath())
        if not ("/left_gripper/" in path or "/right_gripper/" in path):
            continue
        if not collision.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(collision).GetCollisionEnabledAttr().Get() is False:
            continue
        bound, _ = UsdShade.MaterialBindingAPI(collision).ComputeBoundMaterial("physics")
        if bound and str(bound.GetPath()).endswith("/distalGripperPhysicsMaterial"):
            continue
        bind_physics_material(path, link_material_path)
    _apply_task000901_gripper_screw_material(result.GetStage(), str(result.GetPath()))
    return result


def _apply_task000901_gripper_screw_material(stage, robot_path: str) -> None:
    """Bind silver metal only to the authored soft-gripper screw geometry.

    The source CAD stores each distal finger and its exposed screw halves in
    one Mesh without material subsets.  The screw halves are nevertheless
    disconnected topology components (200 faces in the active soft-tip mesh,
    about 12 x 5.5 x 5.5 mm), so make stable face subsets instead of adding
    approximate overlay geometry.
    """

    material_path = f"{robot_path}/Task000901Looks/SilverScrews"
    material_cfg = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.68, 0.71, 0.75),
        metallic=0.95,
        roughness=0.16,
    )
    material_cfg.func(material_path, material_cfg)
    silver = UsdShade.Material(stage.GetPrimAtPath(material_path))
    default = UsdShade.Material(
        stage.GetPrimAtPath(f"{robot_path}/ffw_sg2_follower/Looks/DefaultMaterial")
    )

    subset_count = 0
    distal_mesh_paths = []
    for side_name, side_prefix in (("left", "l"), ("right", "r")):
        for finger in ("r2", "l2"):
            distal_mesh_paths.append(
                f"{robot_path}/ffw_sg2_follower/{side_name}_gripper/"
                f"gripper_{side_prefix}_rh_p12_rn_{finger}/visuals/{finger}/mesh"
            )
    for path in distal_mesh_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
            continue

        mesh = UsdGeom.Mesh(prim)
        counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
        indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
        points = list(mesh.GetPointsAttr().Get() or [])
        if not counts or not points:
            continue

        face_vertices = []
        position_faces = {}
        cursor = 0
        for face_index, count in enumerate(counts):
            vertices = indices[cursor : cursor + count]
            cursor += count
            face_vertices.append(vertices)
            for vertex in vertices:
                point = points[vertex]
                key = tuple(round(float(point[axis]), 4) for axis in range(3))
                position_faces.setdefault(key, []).append(face_index)

        parent = list(range(len(counts)))

        def find(value):
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(first, second):
            first_root, second_root = find(first), find(second)
            if first_root != second_root:
                parent[second_root] = first_root

        for faces in position_faces.values():
            for face in faces[1:]:
                union(faces[0], face)

        components = {}
        for face_index, vertices in enumerate(face_vertices):
            component = components.setdefault(find(face_index), {"faces": [], "vertices": set()})
            component["faces"].append(face_index)
            component["vertices"].update(vertices)

        screw_faces = []
        for component in components.values():
            if len(component["faces"]) not in (200, 224):
                continue
            component_points = [points[index] for index in component["vertices"]]
            extents = []
            for axis in range(3):
                values = [float(point[axis]) for point in component_points]
                extents.append(max(values) - min(values))
            if 10.0 <= extents[0] <= 13.5 and 4.5 <= extents[1] <= 6.0 and 4.5 <= extents[2] <= 6.0:
                screw_faces.extend(component["faces"])

        if not screw_faces:
            continue

        subset = UsdGeom.Subset.Define(stage, f"{path}/Task000901SilverScrews")
        subset.CreateElementTypeAttr(UsdGeom.Tokens.face)
        subset.CreateFamilyNameAttr("materialBind")
        subset.CreateIndicesAttr(sorted(screw_faces))
        UsdGeom.Subset.SetFamilyType(mesh, "materialBind", UsdGeom.Tokens.nonOverlapping)
        # The shared SG2 spawner deliberately makes the body binding strong;
        # relax it here so this task-local face subset can override the screws.
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            default, UsdShade.Tokens.weakerThanDescendants
        )
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(
            silver, UsdShade.Tokens.strongerThanDescendants
        )
        subset_count += 1

    if subset_count != 4:
        # Some real-tip meshes do not preserve the disconnected CAD screw
        # components.  This is a visual-only decoration: keep the inherited
        # black arm material instead of preventing the physics scene from
        # loading.
        print(
            f"[Task901 gripper visual] expected four silver screw subsets below "
            f"{robot_path}, got {subset_count}; keeping the inherited material."
        )
        return
    print(f"[Task901 gripper visual] bound silver metal to {subset_count} screw subsets.")


def make_task000901_sg2_cfg():
    robot = deepcopy(FFW_SG2_CFG)
    robot.prim_path = "{ENV_REGEX_NS}/Robot"
    robot.init_state.pos = SG2_ROOT_POS
    robot.init_state.rot = SG2_ROOT_ROT
    robot.init_state.joint_pos.update(SG2_REFERENCE_JOINT_POS)
    robot.spawn.func = spawn_task000901_test4_robot
    robot.spawn.articulation_props.enabled_self_collisions = True
    robot.spawn.articulation_props.solver_velocity_iteration_count = TEST4_PROFILE[
        "robot_solver_velocity_iterations"
    ]
    return robot


def make_task000901_head_camera_cfg(*, calibrated: bool = False):
    camera = make_ffw_sg2_head_camera_cfg(
        height=720,
        width=1280,
        intrinsic_matrix=TASK000901_SIM_HEAD_K,
    )
    if calibrated:
        camera.class_type = Task000901HeadCamera
        camera.spawn.func = spawn_task000901_zed_head_camera
        # Fit MCAP 20-22 with the head camera 3 mm lower and 0.25 deg higher gaze;
        # refine yaw by 0.1 deg against episodes 20 and 21 gripper visibility;
        # Final: v67 head-camera pose with -0.2 deg local optical-axis roll.
        camera.offset = CameraCfg.OffsetCfg(
            pos=(-0.017737513, 0.024808872, 0.014672518),
            rot=(0.5099947904, 0.4889395455, -0.4956619550, -0.5051362796),
            convention="opengl",
        )
        camera.spawn.clipping_range = (0.15, 8.0)
    return camera


def make_task000901_wrist_camera_cfg(side: str, *, calibrated: bool = False):
    # 1050 publishes 640x480 sideways; the SG2 sim camera emits the same view
    # turned counterclockwise to 480x640. These are the corresponding K values.
    target = TASK000901_SIM_WRIST_K[side]
    camera = make_ffw_sg2_wrist_camera_cfg(
        side,
        width=480,
        height=640,
        intrinsic_matrix=target,
    )
    if calibrated:
        camera.class_type = Task000901WristCamera
        camera.spawn.func = spawn_task000901_d405_wrist_camera
        if side == "left":
            # EP20/21 wrist framing with measured D405 intrinsics retained.
            # Move the optical center 10 mm down-image and 17.5 mm forward,
            # then pitch upward 0.6875 deg to exclude the proximal folds.
            camera.offset = CameraCfg.OffsetCfg(
                pos=(0.010447335, 0.020161260, 0.001632971),
                rot=(-0.013644222, 0.705766523, -0.014226534, -0.708170217),
                convention="opengl",
            )
            camera.spawn.clipping_range = (0.11, 8.0)
        else:
            # Same optical-frame adjustment as the left camera. Preserve
            # the measured intrinsics and existing near clipping distance.
            camera.offset = CameraCfg.OffsetCfg(
                pos=(0.012765638, 0.018971019, -0.000369161),
                rot=(-0.011940489, 0.708077537, -0.009214445, -0.705973597),
                convention="opengl",
            )
            camera.spawn.clipping_range = (0.10, 8.0)
    return camera

def make_task000901_environment_cfg():
    return AssetBaseCfg(prim_path='{ENV_REGEX_NS}/WhiteRoom',spawn=sim_utils.UsdFileCfg(usd_path=ROOM_USD_PATH))

@configclass
class Task000901SceneCfg(InteractiveSceneCfg):
    environment=make_task000901_environment_cfg()
    cucumber=AssetBaseCfg(
        prim_path='{ENV_REGEX_NS}/Cucumber',
        spawn=sim_utils.UsdFileCfg(usd_path=str(PRODUCE_USD_DIR/'cucumber/cucumber.usd')),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.4486041440720182, 0.27804812794732603, PLATE_HOLDER_TABLE_TOP_Z_M+0.0002), rot=(0.9759045155823948, 0.0, 0.0, 0.21819802122359255)),
    )
    onion=AssetBaseCfg(
        prim_path='{ENV_REGEX_NS}/Onion',
        spawn=sim_utils.UsdFileCfg(usd_path=str(PRODUCE_USD_DIR/'onion/onion.usd')),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.39801668764658954, 0.18202717578224217, PLATE_HOLDER_TABLE_TOP_Z_M+0.0002), rot=(0.999997460187401, 0.0, 0.0, -0.002253801582043)),
    )
    pot=AssetBaseCfg(
        prim_path='{ENV_REGEX_NS}/Pot',
        spawn=sim_utils.UsdFileCfg(usd_path=str(COOKWARE_USD_DIR/'pot/pot.usd')),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.5150285698127093, -0.10898928709767158, PLATE_HOLDER_TABLE_TOP_Z_M+0.0002), rot=(0.9691869348185329, 0.0, 0.0, -0.24632637978311778)),
    )
    lid=AssetBaseCfg(
        prim_path='{ENV_REGEX_NS}/Lid',
        spawn=sim_utils.UsdFileCfg(usd_path=str(COOKWARE_USD_DIR/'lid/lid.usd')),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.49755452280610674, -0.35719071783900835, PLATE_HOLDER_TABLE_TOP_Z_M+0.0002), rot=(0.999997460187401, 0.0, 0.0, -0.002253801582043)),
    )
    robot=make_task000901_sg2_cfg()
    cam_head=make_task000901_head_camera_cfg()
    cam_wrist_left=make_task000901_wrist_camera_cfg("left")
    cam_wrist_right=make_task000901_wrist_camera_cfg("right")
    ceiling_spots=AssetBaseCfg(prim_path='{ENV_REGEX_NS}/RailLights',spawn=sim_utils.UsdFileCfg(usd_path=LIGHTING_USD_PATH))
    dome=AssetBaseCfg(prim_path='/World/Dome',spawn=sim_utils.DomeLightCfg(intensity=220))

@configclass
class EmptyCfg:
    pass


@configclass
class ResetEventsCfg:
    reset_scene_to_default = EventTerm(
        func=reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )

@configclass
class Task000901EnvCfg(ManagerBasedRLEnvCfg):
    scene=Task000901SceneCfg(num_envs=1,env_spacing=5.)
    actions=EmptyCfg()
    observations=EmptyCfg()
    rewards=EmptyCfg()
    terminations=EmptyCfg()
    events=ResetEventsCfg()
    decimation=1
    episode_length_s=60.
    def __post_init__(self):
        self.sim.dt=1/60
        self.sim.render.enable_translucency=True
        self.viewer.eye=(1.1,-.40,1.45)
        self.viewer.lookat=(-.55,-.06,.5)


@clone
def spawn_task000901_plate(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Add Task901 contact material and the optional five-strip bending proxy."""
    prim = from_files.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    material_path = f"{prim_path}/PlatePhysicsMaterial"
    # Starting simulation parameters, not measured real friction coefficients.
    material = sim_utils.RigidBodyMaterialCfg(
        static_friction=TEST4_PROFILE["plate_static_friction"],
        dynamic_friction=TEST4_PROFILE["plate_dynamic_friction"], restitution=0.0,
        # Gripper materials use max; the holder uses min. With the plate on
        # min this yields plate/gripper=max and plate/holder=min.
        friction_combine_mode=("max" if os.environ.get("TASK901_PHYSICS_PROFILE") == "test4" else "min"),
        restitution_combine_mode="min",
    )
    material.func(material_path, material)
    bind_physics_material(prim_path, material_path)
    if str(cfg.usd_path).endswith("_flex5.usda"):
        stage = prim.GetStage()
        UsdPhysics.ArticulationRootAPI.Apply(prim)
        articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
        articulation_api.CreateEnabledSelfCollisionsAttr().Set(False)
        articulation_api.CreateSolverPositionIterationCountAttr().Set(32)
        articulation_api.CreateSolverVelocityIterationCountAttr().Set(4)
        for strip in range(5):
            link = prim if strip == 2 else stage.GetPrimAtPath(f"{prim_path}/Strip{strip}")
            rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(link)
            rigid_api.CreateEnableCCDAttr().Set(True)
            rigid_api.CreateMaxDepenetrationVelocityAttr().Set(1.0)
        outer_radius = 0.0925
        for lower_strip in range(4):
            upper_strip = lower_strip + 1
            boundary_x = -outer_radius + 2 * outer_radius * upper_strip / 5
            joint = UsdPhysics.RevoluteJoint.Define(
                stage, f"{prim_path}/FlexJoint{lower_strip}_{upper_strip}"
            )
            body0 = prim_path if lower_strip == 2 else f"{prim_path}/Strip{lower_strip}"
            body1 = prim_path if upper_strip == 2 else f"{prim_path}/Strip{upper_strip}"
            joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
            joint.CreateAxisAttr().Set("Y")
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(boundary_x, 0.0, 0.001))
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(boundary_x, 0.0, 0.001))
            joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
            joint.CreateLowerLimitAttr().Set(-1.25)
            joint.CreateUpperLimitAttr().Set(1.25)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
            drive.CreateTypeAttr().Set("force")
            drive.CreateTargetPositionAttr().Set(0.0)
            drive.CreateStiffnessAttr().Set(0.05)
            drive.CreateDampingAttr().Set(0.01)
    else:
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateEnableCCDAttr().Set(True)
    return prim


def _plate_cfg(name: str, color: str, xy: tuple[float, float]) -> RigidObjectCfg:
    """Use one 15 g rigid plate without the five-strip bending approximation."""
    return RigidObjectCfg(
        prim_path=f'{{ENV_REGEX_NS}}/{name}',
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(PLATE_USD_DIR/f'{color}.usda'),
            func=spawn_task000901_plate,
            activate_contact_sensors=True,
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(xy[0], xy[1], PLATE_HOLDER_TABLE_TOP_Z_M+0.0005),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


@clone
def spawn_task000901_holder(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the fixed wood holder with a low-friction insertion surface."""
    prim = from_files.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    material_path = f"{prim_path}/HolderPhysicsMaterial"
    material = sim_utils.RigidBodyMaterialCfg(
        static_friction=TEST4_PROFILE["holder_static_friction"],
        dynamic_friction=TEST4_PROFILE["holder_dynamic_friction"],
        restitution=0.0,
        friction_combine_mode=("average" if os.environ.get("TASK901_PHYSICS_PROFILE") == "test4" else "min"),
        restitution_combine_mode="min",
    )
    material.func(material_path, material)
    bind_physics_material(prim_path, material_path)
    return prim


@configclass
class Task000901PlateHolderSceneCfg(InteractiveSceneCfg):
    """Measured plate holder and three colored plates in the live 1050 layout."""
    environment=AssetBaseCfg(prim_path='{ENV_REGEX_NS}/WhiteRoom',spawn=sim_utils.UsdFileCfg(usd_path=str(PLATE_HOLDER_ROOM_USD_PATH)))
    # 2026-11-03 episode-0 initial rim fit; motion is left to contact dynamics.
    plate_pink=_plate_cfg('PlatePink', 'pastel_pink', (-0.6490314888984579, -0.3477880991282417))
    plate_blue=_plate_cfg('PlateBlue', 'light_blue', (-0.43551478730348353, -0.21053895671236292))
    plate_yellow=_plate_cfg('PlateYellow', 'lime_yellow', (-0.44920204708130185, -0.443417798818687))
    plate_holder=AssetBaseCfg(
        prim_path='{ENV_REGEX_NS}/PlateHolder',
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(PLATE_HOLDER_USD_PATH),
            func=spawn_task000901_holder,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            # Move 20 mm toward the table rear along table-local +Y
            # (table yaw 89.741733 deg); preserve holder height and yaw.
            pos=(-0.5776103404, 0.1485561007, PLATE_HOLDER_TABLE_TOP_Z_M+0.0022),
            rot=(0.7038995816, 0.0, 0.0, 0.7102994995),
        ),
    )
    robot=make_task000901_sg2_cfg()
    cam_head=make_task000901_head_camera_cfg(calibrated=True)
    cam_wrist_left=make_task000901_wrist_camera_cfg('left', calibrated=True)
    cam_wrist_right=make_task000901_wrist_camera_cfg('right', calibrated=True)
    ceiling_spots=AssetBaseCfg(
        prim_path='{ENV_REGEX_NS}/RailLights',
        spawn=sim_utils.UsdFileCfg(usd_path=LIGHTING_USD_PATH),
    )
    dome=AssetBaseCfg(prim_path='/World/Dome', spawn=sim_utils.DomeLightCfg(intensity=220))


@configclass
class Task000901PlateHolderEnvCfg(Task000901EnvCfg):
    scene=Task000901PlateHolderSceneCfg(num_envs=1, env_spacing=5.)

    def __post_init__(self):
        super().__post_init__()
        if _TASK901_MANUAL_PROFILE is not None:
            _MANUAL_MODULE.apply_config(self, _TASK901_MANUAL_PROFILE)
        # Thin 1 mm plates need finer contact integration than the visual scene.
        self.sim.dt = 1 / 240
        self.sim.physx.enable_ccd = True
        self.decimation = 4
        self.sim.render_interval = self.decimation
