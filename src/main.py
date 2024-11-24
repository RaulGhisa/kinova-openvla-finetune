# coding=utf-8
# Copyright 2024 DeepMind Technologies Limited..
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A simple binary to run catch for a while and record its trajectories.
"""

import time

from absl import app
from absl import flags
from absl import logging
import envlogger
from envlogger.backends import tfds_backend_writer

from envlogger.testing import catch_env
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from environment import Gen3LiteEnv
from mock_node import MockROSNode
from mock_video_capture_device import MockVideoCaptureDevice


FLAGS = flags.FLAGS

flags.DEFINE_integer('num_episodes', 1000, 'Number of episodes to log.')
flags.DEFINE_string('trajectories_dir', '/home/psxrg6/Documents/code/kinova-openvla/trajectories_dir',
                    'Path in a filesystem to record trajectories.')


def record_data(unused_argv):
    logging.info('Creating Catch environment...')

    node = MockROSNode()
    while not node.is_ready.is_set():
        time.sleep(0.1)

    env = Gen3LiteEnv(node=node,
                      camera_device=MockVideoCaptureDevice())
    logging.info('Done creating Catch environment.')

    def step_fn(unused_timestep, unused_action, unused_env):
        return {'timestamp': time.time()}

    dataset_config = tfds.rlds.rlds_base.DatasetConfig(
        name='kinova_example',
        observation_info=tfds.features.FeaturesDict({
            'observation': tfds.features.FeaturesDict({
                'image': tfds.features.Image(
                    shape=(256, 256, 3),
                    dtype=np.uint8,
                    encoding_format='jpeg',
                    doc='Main camera RGB observation.',
                ),
                'wrist_image': tfds.features.Image(
                    shape=(256, 256, 3),
                    dtype=np.uint8,
                    encoding_format='jpeg',
                    doc='Wrist camera RGB observation.',
                ),
                'state': tfds.features.Tensor(
                    shape=(8,),
                    dtype=np.float32,
                    doc='Robot EEF state (6D pose, 2D gripper).',
                ),
                'joint_state': tfds.features.Tensor(
                    shape=(6,),
                    dtype=np.float32,
                    doc='Robot joint angles.',
                ),
                'is_first': tfds.features.Scalar(
                    # shape=(1,),
                    dtype=np.bool_,
                    doc='True on first step of the episode.'
                ),
                'is_last': tfds.features.Scalar(
                    # shape=(1,),
                    dtype=np.bool_,
                    doc='True on last step of the episode.'
                ),
                'is_terminal': tfds.features.Scalar(
                    # shape=(1,),
                    dtype=np.bool_,
                    doc='True on last step of the episode if it is a terminal step, True for demos.'
                ),
                'language_instruction': tfds.features.Text(
                    doc='Language Instruction.'
                )
            })
        }),
        reward_info=tf.float64,
        discount_info=tf.float64,
        action_info=tfds.features.Tensor(
            shape=(7,),
            dtype=np.float64,
            doc='Action.',
        ),
        step_metadata_info={'timestamp': tf.float32})

    logging.info('Wrapping environment with EnvironmentLogger...')
    with envlogger.EnvLogger(
        env,
        step_fn=step_fn,
        backend=tfds_backend_writer.TFDSBackendWriter(
            data_directory=FLAGS.trajectories_dir,
            split_name='train',
            max_episodes_per_file=500,
            ds_config=dataset_config),
    ) as env:
        logging.info('Done wrapping environment with EnvironmentLogger.')

        logging.info('Training a random agent for %r episodes...',
                     FLAGS.num_episodes)

        timestep = env.reset()
        while not timestep.last():
            action = np.random.rand(7)
            timestep = env.step(action)

        logging.info('Done training a random agent for %r episodes.',
                     FLAGS.num_episodes)


if __name__ == '__main__':
    app.run(record_data)
