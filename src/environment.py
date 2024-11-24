import logging
import dm_env
from dm_env import specs
import numpy as np

from data_types import VideoCaptureDevice
from mock_node import ROSNode


class Gen3LiteEnv(dm_env.Environment):
    def __init__(self, node: ROSNode, camera_device: VideoCaptureDevice):
        self._node = node
        self._camera_device = camera_device
        self._instruction = "pick up a banana"  # pick one instruction for now
        self._step_count = 0

    def reset(self):
        """ Resets the environment state. """

        logging.info("Resetting environment...")

        self._step_count = 0
        return dm_env.restart(
            self._get_observation_as_dict(),
        )

    def _get_observation_as_dict(self):
        image = self._camera_device.get_camera_image()
        joint_state, eef_state = self._node.get_joint_and_eef_state()

        # clone image to keep data type consistency, optional anyway
        wrist_image = image[:]

        observation = {
            'observation': {
                'image': image,
                'wrist_image': wrist_image,
                'state': np.asarray(eef_state.to_list(), dtype=np.float32),
                'joint_state': np.asarray(joint_state.to_list(), dtype=np.float32),
                'is_first': self._step_count == 0,
                'is_last': self._step_count >= 10,
                'is_terminal': self._step_count >= 10,
                'language_instruction': self._instruction,
            },
        }

        return observation

    def step(self, action):
        """
            Apply the action and update the environment state
            The action is delta motion.
        """
        logging.info(f"Taking action {action}...")

        self._step_count += 1
        # perform action and wait for it to finish...
        self._node.perform_action(action)
        reward = 0.0

        is_done = self._step_count >= 10

        if is_done:
            return dm_env.termination(reward, self._get_observation_as_dict())
        return dm_env.transition(reward, self._get_observation_as_dict())

    def action_spec(self):
        """ 
            Should be 7 values: displacement of translation, rotation and gripper.
        """
        return specs.Array(
            shape=(7,),
            dtype=np.float32,
            name='action'
        )

    def observation_spec(self) -> dict[str, specs.Array]:
        """
            Returns the observation spec following RLDS standard.
            The observation spec should be a nested dictionary where each leaf node is a dm_env.specs. Array describing the shape and type of the observation. 
            Source: Claude.ai
        """
        return {
            'observation': {
                'image': specs.Array(
                    shape=(256, 256, 3),
                    dtype=np.uint8,
                ),
                'wrist_image': specs.Array(
                    shape=(256, 256, 3),
                    dtype=np.uint8,
                ),
                'state': specs.Array(
                    shape=(8,),
                    dtype=np.float32,
                ),
                'joint_state': specs.Array(
                    shape=(7,),
                    dtype=np.float32,
                ),
                'is_first': specs.Array(
                    shape=(),
                    dtype=np.bool_,
                ),
                'is_last': specs.Array(
                    shape=(),
                    dtype=np.bool_,
                ),
                'is_terminal': specs.Array(
                    shape=(),
                    dtype=np.bool_,
                ),
                'language_instruction': specs.StringArray(shape=(100,)),
            }
        }


def main(args=None):
    rclpy.init(args=args)
    gen3_lite_env = Gen3LiteEnv()

    executor = MultiThreadedExecutor()
    executor.add_node(gen3_lite_env.node)
    try:
        # rclpy.spin(gen3_lite_env.node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        gen3_lite_env.node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
