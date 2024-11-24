import tensorflow_datasets as tfds

file = tfds.builder_from_directory(
    '/home/psxrg6/Documents/model-00004-of-00004.safetensors').as_dataset(split='all')

pass
