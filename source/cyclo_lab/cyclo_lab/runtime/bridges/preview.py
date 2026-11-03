"""Read-only ROS preview for environments without the policy action contract."""

from __future__ import annotations

import threading

import torch
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

from cyclo_lab.robot_specs.ffw.sg2 import (
    BASE_BODY, BASE_FRAME, FFW_SG2_CAMERA_TOPICS, FFW_SG2_PUBLISHED_JOINT_NAMES,
    JOINT_STATES_TOPIC, SIMULATION_RESET_TOPIC,
)
from cyclo_lab.runtime.publishers.articulation_state_publisher import ArticulationStatePublisher
from cyclo_lab.runtime.publishers.camera_publishers import CompressedCameraPublishers
from cyclo_lab.runtime.transport.ros2_zenoh import EMPTY, close_endpoints, create_subscriber


class PreviewTopicBridge:
    """Publish cameras and joint states; never subscribe to policy commands."""

    requires_activation = False
    accepts_policy_commands = False

    def __init__(self, env, *, camera_publish_hz: float, idle_action=None) -> None:
        self.env = env
        self._reset_requested = threading.Event()
        self._action = torch.zeros(
            (env.num_envs, env.action_manager.total_action_dim),
            device=env.device, dtype=torch.float32,
        )
        if idle_action is not None and len(idle_action) == env.action_manager.total_action_dim:
            self._action[:] = torch.as_tensor(idle_action, device=env.device, dtype=torch.float32)
        self._hold_current_joint_actions()

        camera_topics = {
            name: topic for name, topic in FFW_SG2_CAMERA_TOPICS.items()
            if name in env.scene.sensors
        }
        if not camera_topics:
            first_camera = next((
                name for name, sensor in env.scene.sensors.items()
                if "rgb" in getattr(getattr(sensor, "data", None), "output", {})
            ), None)
            if first_camera is not None:
                camera_topics[first_camera] = FFW_SG2_CAMERA_TOPICS["cam_head"]
        if not camera_topics:
            raise ValueError("Environment has no RGB camera for UI preview")
        self.camera_publishers = CompressedCameraPublishers(
            env.scene, camera_topics, camera_publish_hz,
        )
        self.state_publisher = ArticulationStatePublisher(
            env.scene["robot"], joint_names=FFW_SG2_PUBLISHED_JOINT_NAMES,
            joint_states_topic=JOINT_STATES_TOPIC, base_frame=BASE_FRAME,
            base_body=BASE_BODY,
        )
        self._subscriber = create_subscriber(
            topic=SIMULATION_RESET_TOPIC, msg_type=EMPTY,
            callback=lambda _message: self._reset_requested.set(),
        )

    def consume_reset_request(self) -> bool:
        requested = self._reset_requested.is_set()
        self._reset_requested.clear()
        return requested

    def reset(self) -> None:
        self._reset_requested.clear()
        self._hold_current_joint_actions()

    def _hold_current_joint_actions(self) -> None:
        """Keep absolute joint terms at reset pose; leave relative/IK terms idle."""

        offset = 0
        for term in self.env.action_manager._terms.values():
            if isinstance(term, JointPositionAction):
                positions = term._asset.data.joint_pos[:, term._joint_ids]
                self._action[:, offset:offset + term.action_dim] = (
                    positions - term._offset
                ) / term._scale
            offset += term.action_dim

    def get_action(self):
        return self._action

    def get_hold_action(self):
        return self._action

    def publish_observations(self) -> bool:
        self.state_publisher.publish_joint_states()
        return self.camera_publishers.publish()

    def close(self) -> None:
        self.state_publisher.close()
        close_endpoints((*self.camera_publishers.endpoints, self._subscriber))
