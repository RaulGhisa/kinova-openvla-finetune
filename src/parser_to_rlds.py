def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            # this is a list of dicts in our case
            data = np.load(episode_path, allow_pickle=True)

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i, step in enumerate(data):
                # compute Kona language embedding
                language_embedding = self._embed(
                    [step['language_instruction']])[0].numpy()

                episode.append({
                    'observation': {
                        'image': step['image'],
                        'wrist_image': step['wrist_image'],
                        'state': step['state'],
                    },
                    'action': step['action'],
                    'discount': 1.0,
                    'reward': float(i == (len(data) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(data) - 1),
                    'is_terminal': i == (len(data) - 1),
                    'language_instruction': step['language_instruction'],
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)