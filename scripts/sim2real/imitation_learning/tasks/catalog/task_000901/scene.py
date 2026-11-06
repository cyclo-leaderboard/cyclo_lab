"""Task000901 real-aligned WhiteRoom scene with the SG2 reference pose."""
from copy import deepcopy
from pathlib import Path
import torch
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import from_files
from isaaclab.sim.utils import bind_physics_material, clone
from pxr import PhysxSchema
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.events import reset_scene_to_default
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from cyclo_lab.assets.robots.FFW_SG2 import FFW_SG2_CFG
from cyclo_lab.assets.sensors.ffw_sg2_cameras import (
    make_ffw_sg2_head_camera_cfg,
    make_ffw_sg2_wrist_camera_cfg,
)
_ROOT=Path(__file__).resolve().parent
TABLE_TOP_Z_M=.70
ROOM_USD_PATH=str(_ROOT/'runtime/data/white_room_task000901_visual_v2.usda')
LIGHTING_USD_PATH=str(_ROOT/'runtime/data/lighting_task000901_visual_v2.usda')
PRODUCE_USD_DIR=_ROOT/'runtime/data/objects/visual_v2/produce'
COOKWARE_USD_DIR=_ROOT/'runtime/data/objects/visual_v2/cookware'
PLATE_USD_DIR=_ROOT/'runtime/data/objects/plates_v1'
PLATE_HOLDER_USD_PATH=_ROOT/'runtime/data/objects/plate_holder_v1/wood_plate_holder.usda'
PLATE_HOLDER_ROOM_USD_PATH=_ROOT/'runtime/data/white_room_plate_holder.usda'
PLATE_HOLDER_LIGHTING_USD_PATH=_ROOT/'runtime/data/white_room_plate_holder_lighting.usda'

# Live 1050 SG2 state recaptured 2026-11-02. Wheel encoder positions are not
# copied because they do not affect the static alignment view.
SG2_REFERENCE_JOINT_POS = {
    "arm_l_joint1": 0.9999996788,
    "arm_l_joint2": 0.0,
    "arm_l_joint3": 0.0,
    "arm_l_joint4": -2.1999801186,
    "arm_l_joint5": 0.0,
    "arm_l_joint6": 0.0,
    "arm_l_joint7": 0.0,
    "arm_r_joint1": 0.9999876946,
    "arm_r_joint2": 0.0,
    "arm_r_joint3": 0.0,
    "arm_r_joint4": -2.1999921028,
    "arm_r_joint5": 0.0,
    "arm_r_joint6": 0.0,
    "arm_r_joint7": 0.0,
    "gripper_l_joint1": 0.0,
    "gripper_r_joint1": 0.0,
    "head_joint1": 0.8298836062,
    "head_joint2": 0.0,
    "lift_joint": -0.2,
}

# The tabletop stays the task coordinate anchor. The root stays on the floor.
SG2_ROOT_POS = (-0.0200190219, -0.1099593887, 0.0)
SG2_ROOT_ROT = (0.0, 0.0, 0.0, 1.0)


class Task000901WristCamera(Camera):
    """Apply the live RealSense color response at simulated sensor readout."""

    def _update_buffers_impl(self, env_ids):
        super()._update_buffers_impl(env_ids)
        rgb = self._data.output.get("rgb")
        if rgb is None:
            return
        if "wrist_left" in self.cfg.prim_path:
            # A smooth exposure/tone response retains shadow detail. The old
            # negative black levels clipped the tabletop shadows to black.
            gamma = torch.tensor((1.35, 1.20, 1.05), device=rgb.device)
            signal = rgb[env_ids].to(torch.float32) / 255.0
            gain = torch.tensor((0.97, 0.92, 0.96), device=rgb.device)
            corrected = 255.0 * gain * signal.pow(gamma)
            # Preserve the bright plate/AprilTag highlights while matching
            # the measured wall and table midtones. Keep shadow detail intact.
            highlight = ((signal - 0.72) / 0.22).clamp(0, 1).square()
            corrected = corrected + highlight * (255.0 - corrected)
            luma = (corrected * torch.tensor((0.2126, 0.7152, 0.0722), device=rgb.device)).sum(dim=-1, keepdim=True)
            corrected = luma + 1.08 * (corrected - luma)
            rgb[env_ids] = corrected.clamp_(0, 255).to(rgb.dtype)
            return
        gain, bias = (0.94, 0.92, 1.06), (0.0, 0.0, 0.0)
        scale = torch.tensor(gain, device=rgb.device, dtype=torch.float32)
        black = torch.tensor(bias, device=rgb.device, dtype=torch.float32)
        corrected = rgb[env_ids].to(torch.float32) * scale + black
        # Reduce the yellow cast of warm surfaces in the right wrist sensor.
        # The blue wall/table response is unaffected by this chroma term.
        yellow = (corrected[..., 1] - corrected[..., 2]).clamp_min(0)
        corrected[..., 2] += 0.55 * yellow
        rgb[env_ids] = corrected.clamp_(0, 255).to(rgb.dtype)


def make_task000901_sg2_cfg():
    robot = deepcopy(FFW_SG2_CFG)
    robot.prim_path = "{ENV_REGEX_NS}/Robot"
    robot.init_state.pos = SG2_ROOT_POS
    robot.init_state.rot = SG2_ROOT_ROT
    robot.init_state.joint_pos.update(SG2_REFERENCE_JOINT_POS)
    return robot


def make_task000901_head_camera_cfg(*, calibrated: bool = False):
    camera = make_ffw_sg2_head_camera_cfg(
        height=720,
        width=1280,
        intrinsic_matrix=(
            734.6712036132812, 0.0, 626.33935546875,
            0.0, 734.6712036132812, 360.0394592285156,
            0.0, 0.0, 1.0,
        ),
    )
    if calibrated:
        # Fit the table rear edge and visible gripper tips in the head frame.
        camera.spawn.focal_length *= 0.96
        camera.offset = CameraCfg.OffsetCfg(
            pos=(0.0, 0.03, 0.0),
            rot=(0.5021662335, 0.4978029242, -0.5039221908, -0.4960682899),
            convention="opengl",
        )
    return camera


def make_task000901_wrist_camera_cfg(side: str, *, calibrated: bool = False):
    # 1050 publishes 640x480 sideways; the SG2 sim camera emits the same view
    # turned counterclockwise to 480x640. These are the corresponding K values.
    if side == "left":
        fx, fy, cx, cy = 392.2402038574219, 392.85992431640625, 242.0066680908203, 317.3694152832031
    else:
        fx, fy, cx, cy = 391.906982421875, 392.4100646972656, 240.0281982421875, 323.3479919433594
    camera = make_ffw_sg2_wrist_camera_cfg(
        side,
        width=480,
        height=640,
        intrinsic_matrix=(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0),
    )
    if calibrated:
        camera.class_type = Task000901WristCamera
    return camera

def make_task000901_environment_cfg():
    return AssetBaseCfg(prim_path='{ENV_REGEX_NS}/WhiteRoom',spawn=sim_utils.UsdFileCfg(usd_path=ROOM_USD_PATH))

@configclass
class Task000901SceneCfg(InteractiveSceneCfg):
    environment=make_task000901_environment_cfg()
    cucumber=AssetBaseCfg(
        prim_path='{ENV_REGEX_NS}/Cucumber',
        spawn=sim_utils.UsdFileCfg(usd_path=str(PRODUCE_USD_DIR/'cucumber/cucumber.usd')),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.47020769846105787, 0.2765905432369996, 0.7002), rot=(0.9754102616434104, 0.0, 0.0, 0.22039696341087303)),
    )
    onion=AssetBaseCfg(
        prim_path='{ENV_REGEX_NS}/Onion',
        spawn=sim_utils.UsdFileCfg(usd_path=str(PRODUCE_USD_DIR/'onion/onion.usd')),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.4191879324760945, 0.18079859430013154, 0.7002), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    pot=AssetBaseCfg(
        prim_path='{ENV_REGEX_NS}/Pot',
        spawn=sim_utils.UsdFileCfg(usd_path=str(COOKWARE_USD_DIR/'pot/pot.usd')),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.5348868417584767, -0.11074235415490319, 0.7002), rot=(0.969739644355927, 0.0, 0.0, -0.24414139789155043)),
    )
    lid=AssetBaseCfg(
        prim_path='{ENV_REGEX_NS}/Lid',
        spawn=sim_utils.UsdFileCfg(usd_path=str(COOKWARE_USD_DIR/'lid/lid.usd')),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.5162941809369388, -0.3588624974411967, 0.7002), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    robot=make_task000901_sg2_cfg()
    cam_head=make_task000901_head_camera_cfg()
    cam_wrist_left=make_task000901_wrist_camera_cfg("left")
    cam_wrist_right=make_task000901_wrist_camera_cfg("right")
    ceiling_spots=AssetBaseCfg(prim_path='{ENV_REGEX_NS}/RailLights',spawn=sim_utils.UsdFileCfg(usd_path=LIGHTING_USD_PATH))
    dome=AssetBaseCfg(prim_path='/World/Dome',spawn=sim_utils.DomeLightCfg(intensity=400))

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
    """Keep the thin concave profile; add plate-specific contact material and CCD."""
    prim = from_files.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    material_path = f"{prim_path}/PlatePhysicsMaterial"
    # Starting simulation parameters, not measured real friction coefficients.
    material = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.6, dynamic_friction=0.45, restitution=0.0,
        friction_combine_mode="max", restitution_combine_mode="min",
    )
    material.func(material_path, material)
    bind_physics_material(prim_path, material_path)
    PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateEnableCCDAttr().Set(True)
    return prim


def _plate_cfg(name: str, color: str, xy: tuple[float, float]) -> RigidObjectCfg:
    """Share Task901's 1.0 mm plate profile and convex sectors through a color-only USD layer."""
    return RigidObjectCfg(
        prim_path=f'{{ENV_REGEX_NS}}/{name}',
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(PLATE_USD_DIR/f'{color}.usda'),
            func=spawn_task000901_plate,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, kinematic_enabled=False, disable_gravity=False,
                solver_position_iteration_count=32, solver_velocity_iteration_count=4,
                max_depenetration_velocity=1.0, linear_damping=0.01, angular_damping=0.05,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.015),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(xy[0], xy[1], TABLE_TOP_Z_M+0.0005),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


@configclass
class Task000901PlateHolderSceneCfg(InteractiveSceneCfg):
    """Measured plate holder and three colored plates in the live 1050 layout."""
    environment=AssetBaseCfg(prim_path='{ENV_REGEX_NS}/WhiteRoom',spawn=sim_utils.UsdFileCfg(usd_path=str(PLATE_HOLDER_ROOM_USD_PATH)))
    # 2026-11-03 episode-0 initial rim fit; motion is left to contact dynamics.
    plate_pink=_plate_cfg('PlatePink', 'pastel_pink', (-0.6679749960028153, -0.3500986706913904))
    plate_blue=_plate_cfg('PlateBlue', 'light_blue', (-0.4550791270214076, -0.21188847597138133))
    plate_yellow=_plate_cfg('PlateYellow', 'lime_yellow', (-0.46771652442083467, -0.44482664881554324))
    plate_holder=AssetBaseCfg(
        prim_path='{ENV_REGEX_NS}/PlateHolder',
        spawn=sim_utils.UsdFileCfg(usd_path=str(PLATE_HOLDER_USD_PATH)),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.5464056504055486, 0.15655533467208427, TABLE_TOP_Z_M+0.0002),
            rot=(0.7006398248933365, 0.0, 0.0, 0.7135151265204086),
        ),
    )
    robot=make_task000901_sg2_cfg()
    cam_head=make_task000901_head_camera_cfg(calibrated=True)
    cam_wrist_left=make_task000901_wrist_camera_cfg('left', calibrated=True)
    cam_wrist_right=make_task000901_wrist_camera_cfg('right', calibrated=True)
    ceiling_spots=AssetBaseCfg(
        prim_path='{ENV_REGEX_NS}/RailLights',
        spawn=sim_utils.UsdFileCfg(usd_path=str(PLATE_HOLDER_LIGHTING_USD_PATH)),
    )
    dome=AssetBaseCfg(prim_path='/World/Dome', spawn=sim_utils.DomeLightCfg(intensity=250))


@configclass
class Task000901PlateHolderEnvCfg(Task000901EnvCfg):
    scene=Task000901PlateHolderSceneCfg(num_envs=1, env_spacing=5.)

    def __post_init__(self):
        super().__post_init__()
        # Thin 2 mm plates need finer contact integration than the visual scene.
        self.sim.dt = 1 / 240
        self.sim.physx.enable_ccd = True
        self.decimation = 4
        self.sim.render_interval = self.decimation

