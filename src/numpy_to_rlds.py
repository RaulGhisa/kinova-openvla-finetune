"""
    Courtesy of Claude.ai
"""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def create_rlds_dataset(
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    episode_metadata: dict = None
):
    """
    Creates an RLDS dataset from NumPy arrays.

    Args:
        observations: Array of shape [num_steps, *obs_shape]
        actions: Array of shape [num_steps, *action_shape]
        rewards: Array of shape [num_steps]
        dones: Array of shape [num_steps]
        episode_metadata: Optional dictionary of episode-level metadata

    Returns:
        tf.data.Dataset in RLDS format
    """
    assert len(observations) == len(actions) == len(
        rewards) == len(dones), "All arrays must have same length"

    # Split into episodes based on done signals
    episode_indices = np.where(dones)[0].tolist()
    if not dones[-1]:
        episode_indices.append(len(dones))

    start_idx = 0
    episodes = []

    for end_idx in episode_indices:
        # Create episode dict
        episode = {
            'observation': observations[start_idx:end_idx + 1],
            'action': actions[start_idx:end_idx + 1],
            'reward': rewards[start_idx:end_idx + 1],
            'is_terminal': dones[start_idx:end_idx + 1],
            'is_first': np.zeros(end_idx - start_idx + 1, dtype=bool),
            'is_last': np.zeros(end_idx - start_idx + 1, dtype=bool),
        }

        # Set first and last step flags
        episode['is_first'][0] = True
        episode['is_last'][-1] = True

        # Add episode metadata if provided
        if episode_metadata:
            episode.update(episode_metadata)

        episodes.append(episode)
        start_idx = end_idx + 1

    # Convert episodes to tf.data.Dataset
    def episode_generator():
        for episode in episodes:
            yield episode

    # Define the feature structure
    feature_structure = {
        'observation': tf.TensorSpec(shape=(None,) + observations.shape[1:], dtype=tf.float32),
        'action': tf.TensorSpec(shape=(None,) + actions.shape[1:], dtype=tf.float32),
        'reward': tf.TensorSpec(shape=(None,), dtype=tf.float32),
        'is_terminal': tf.TensorSpec(shape=(None,), dtype=tf.bool),
        'is_first': tf.TensorSpec(shape=(None,), dtype=tf.bool),
        'is_last': tf.TensorSpec(shape=(None,), dtype=tf.bool),
    }

    # Add metadata features if present
    if episode_metadata:
        for key, value in episode_metadata.items():
            if isinstance(value, (int, np.integer)):
                dtype = tf.int64
            elif isinstance(value, (float, np.floating)):
                dtype = tf.float32
            else:
                dtype = tf.string
            feature_structure[key] = tf.TensorSpec(shape=(), dtype=dtype)

    # Create dataset
    dataset = tf.data.Dataset.from_generator(
        episode_generator,
        output_signature=feature_structure
    )

    return dataset


# Example usage
if __name__ == "__main__":
    # Create sample data
    num_steps = 1000
    obs_dim = 4
    action_dim = 2

    observations = np.random.randn(num_steps, obs_dim)
    actions = np.random.randn(num_steps, action_dim)
    rewards = np.random.randn(num_steps)
    dones = np.zeros(num_steps, dtype=bool)
    dones[499] = True  # End of first episode
    dones[-1] = True   # End of second episode

    metadata = {
        'episode_id': 0,
        'difficulty': 'medium',
        'score': 100.0
    }

    # Create RLDS dataset
    dataset = create_rlds_dataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
        episode_metadata=metadata
    )

    # Verify the dataset
    for episode in dataset.take(1):
        print("Episode structure:", {k: v.shape for k, v in episode.items()})
