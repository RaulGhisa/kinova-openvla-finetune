
from dataclasses import dataclass


@dataclass
class EndEffectorState():
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    roll: float
    gripper_x: float
    gripper_y: float

    def to_list(self):
        return [self.x, self.y, self.z, self.yaw, self.pitch, self.roll, self.gripper_x, self.gripper_y]


@dataclass
class JointState():
    j_1: float
    j_2: float
    j_3: float
    j_4: float
    j_5: float
    j_6: float

    def to_list(self):
        return [self.j_1, self.j_2, self.j_3, self.j_4, self.j_5, self.j_6]


class ROSNode():
    """ 
        The base class for ROS nodes to help with type hinting. 
        Extended by MockROSNode as well.
    """

    def _joint_callback(self, msg):
        raise NotImplementedError

    def get_joint_and_eef_state(self) -> tuple[JointState, EndEffectorState]:
        raise NotImplementedError

    def _compute_end_effector_state_from_joints(self, joint_states):
        raise NotImplementedError

    def perform_action(self, action) -> bool:
        raise NotImplementedError


class VideoCaptureDevice():
    """ 
        The base class for an external video capturing devices (not part of a ROS node) to help with type hinting.
        Extended by MockVideoCaptureDevice as well. 
    """
    def get_camera_image():
        raise NotImplementedError
