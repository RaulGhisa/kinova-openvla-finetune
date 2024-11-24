
from dataclasses import dataclass


@dataclass
class EndEffectorState():
    """
        Models end-effector and gripper states. I believe OpenVLA expects an N-dim array with those two.
        Gripper state is unkown...
        More info here: https://github.com/moojink/rlds_dataset_builder/blob/4bfb8af6d4ca5c173703771d8ad3e2e0c980d525/LIBERO_Spatial/LIBERO_Spatial_dataset_builder.py#L116
    """
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
        """
            Updates the driver internal joint state from ROS messages.
        """
        raise NotImplementedError

    def get_joint_and_eef_state(self) -> tuple[JointState, EndEffectorState]:
        """ 
            Returns both the joint and EEF states at the same time.
            EEF state is computed from joints using forward kinematics, this ensures both values match perfectly. 
        """
        raise NotImplementedError

    def _compute_end_effector_state_from_joints(self, joint_states):
        """
            Performs forward kinematics from joint state.
        """
        raise NotImplementedError

    def perform_action(self, action) -> bool:
        """"
            Executes the action coming from Kinova arm driver.
        """
        raise NotImplementedError


class VideoCaptureDevice():
    """ 
        The base class for an external video capturing devices (not part of a ROS node) to help with type hinting.
        Extended by MockVideoCaptureDevice as well. 
    """
    def get_camera_image():
        raise NotImplementedError
