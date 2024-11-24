from copy import deepcopy
import threading
import time
import numpy as np

from data_types import EndEffectorState, JointState, ROSNode
import mock_data


class MockROSNode(ROSNode):
    def __init__(self):
        # cb_group_0 = MutuallyExclusiveCallbackGroup()
        # cb_group_1 = MutuallyExclusiveCallbackGroup()

        # # Create subscription
        # self.joint_sub = self.node.create_subscription(
        #     JointState, "/joint_states", self.joint_callback, 9, callback_group=cb_group_1
        # )
        # # self.tf_sub = self.node.create_subscription(
        # #     TFMessage, "/tf", self.tf_callback, 9
        # # )

        # self.fk_client = self.node.create_client(
        #     GetPositionFK, '/compute_fk', callback_group=cb_group_1)
        # # while not self.fk_client.wait_for_service(timeout_sec=0.0):
        # #     print('Service not available, waiting again...')

        self._current_observation: None | dict = None
        self._joint_states: None | JointState = None
        ee_position: None | EndEffectorState = None
        self._access_lock = threading.RLock()
        self.is_ready = threading.Event()

        threading.Thread(target=self._run,
                         name='mock_ros_node', daemon=True).start()

    def get_joint_and_eef_state(self):
        with self._access_lock:
            joint_states = deepcopy(self._joint_states)

        eef_states = self._compute_end_effector_state_from_joints(
            joint_states=joint_states)
        return joint_states, eef_states

    def _run(self):
        while True:
            self._joint_callback()
            time.sleep(1)

    def _joint_callback(self, msg=mock_data.get_mock_joint_state()):
        with self._access_lock:
            # update joint states from ROS message
            self._joint_states = JointState(
                j_1=msg[0], j_2=msg[1], j_3=msg[2],
                j_4=msg[3], j_5=msg[4], j_6=msg[5]
            )
        self.is_ready.set()

    def _compute_end_effector_state_from_joints(self, joint_states):
        mocks = mock_data.get_mock_state()
        mock_gripper = mock_data.get_mock_gripper_state()
        ee_position = EndEffectorState(
            x=mocks[0], y=mocks[1], z=mocks[2],
            yaw=mocks[3], pitch=mocks[4], roll=mocks[5],
            gripper_x=mock_gripper[0], gripper_y=mock_gripper[1],
        )
        return ee_position

    def perform_action(self, action):
        time.sleep(0.01)
        return True


if __name__ == "__main__":
    MockROSNode()
    while True:
        time.sleep(1)
