import numpy as np


def get_mock_image(h=256, w=256):
    """ Main camera RGB observation. """
    return (255 * np.random.rand(h, w, 3)).astype(dtype=np.uint8)


def get_mock_state():
    """ Robot EEF state 6D pose. """
    return np.random.rand(6,).astype(dtype=np.float32)


def get_mock_gripper_state():
    """ EEF gripper state 2D. """
    return np.random.rand(2,).astype(dtype=np.float32)


def get_mock_joint_state():
    """ Robot joint angles. """
    return np.random.rand(7,).astype(dtype=np.float32)


def get_mock_action():
    """ Robot EEF action. """
    return np.random.rand(7,).astype(dtype=np.float32)


def get_mock_discount():
    """ Discount if provided, default to 1. """
    return float(1.0)


def get_mock_reward():
    """ Reward if provided, 1 on final step for demos. """
    return float(0.0)


def get_mock_is_first():
    """ True on first step of the episode. """
    return False


def get_mock_is_last():
    """ True on last step of the episode. """
    return False


def get_mock_is_terminal():
    """ True on last step of the episode if it is a terminal step, True for demos. """
    return False


def get_mock_language_instruction():
    """ Language Instruction. """
    return 'grab ball and place it in the basket'


def get_mock_episode_metadata():
    """ Path to the original data file. """
    return r'this/is/the/path/to/the/metadata'


def get_mock_step_data():
    return {
        'image': get_mock_image(),
        'wrist_image': get_mock_image(),
        'states': get_mock_state(),
        'joint_states': get_mock_joint_state(),
        'action': get_mock_action(),
        'language_instruction': get_mock_language_instruction(),
    }


def get_mock_episode(n=100):
    episode_data = []

    for _ in range(n):
        episode_data.append(get_mock_step_data())

    return episode_data


if __name__ == "__main__":
    print(get_mock_step_data())
    print(get_mock_episode(1)[0])
