"""my_dataset dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import os


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for my_dataset dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    """my_dataset dataset."""
    import json

    import tensorflow_datasets as tfds
    import tensorflow as tf
    import os
    import cv2
    import numpy as np

    from typing import List

    class Builder(tfds.core.GeneratorBasedBuilder):
        """
          自定義的資料集。
          該資料集包含手部圖片、遮罩和標註資料，從遠端伺服器下載後使用。
      """

        VERSION = tfds.core.Version('1.0.0')
        RELEASE_NOTES = {
            '1.0.0': 'Initial release.',
        }

        def _info(self) -> tfds.core.DatasetInfo:
            """Returns the dataset metadata."""
            # TODO(my_dataset): Specifies the tfds.core.DatasetInfo object

            return self.dataset_info_from_configs(
                features=tfds.features.FeaturesDict({
                    # These are the features of your dataset like images, labels ...
                    'image': tfds.features.Image(shape=(400, 640, 3)),
                    'segmentation': tfds.features.Tensor(shape=(400, 640, 1), dtype=tf.uint8),
                }),

                # homepage='https://dataset-homepage/',
                citation="""\
                        @article{my_dataset,
                          title={CRI CoreHand Datasets},
                          author={Lucas, Frank},
                          journal={Journal of My Dataset},
                          year={2023},
                        }
                      """,
            )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager) -> List[tfds.core.SplitGenerator]:

        """Splits the dataset into training and validation sets.

        Args:
            dl_manager (tfds.download.DownloadManager): The download manager.

        Returns:
            A list of `tfds.core.SplitGenerator` objects for generating the training and validation sets.
        """

        # Set the remote server name, username, password, and file paths
        server_name = '192.168.71.209'
        user_name = 'LucasA220619'
        password = 'Vm35p4ru8 ~'

        remote_path = '/CT-425/Lucas/hand/images.zip'
        remote_mask_path = '/CT-425/Lucas/hand/mask.zip'
        remote_anno_path = '/CT-425/Lucas/hand/annotation.zip'

        # Use the download manager to download and extract the files and get the local paths
        local_dir = dl_manager.download_and_extract(
            [
                f'ftp://{user_name}:{password}@{server_name}{remote_path}',
                f'ftp://{user_name}:{password}@{server_name}{remote_mask_path}',
                f'ftp://{user_name}:{password}@{server_name}{remote_anno_path}'
            ]
        )

        # Get the file paths for all images, masks, and annotations in the dataset
        img_paths = []
        mask_paths = []
        anno_paths = []
        for dirpath, _, filenames in tf.io.gfile.walk(local_dir[0]):
            for name in filenames:
                if name.endswith('.png'):
                    img_paths.append(os.path.join(dirpath, name))

        for dirpath, _, filenames in tf.io.gfile.walk(local_dir[1]):
            for name in filenames:
                if name.endswith('.npy'):
                    mask_paths.append(os.path.join(dirpath, name))

        for dirpath, _, filenames in tf.io.gfile.walk(local_dir[2]):
            for name in filenames:
                if name.endswith('.json'):
                    anno_paths.append(os.path.join(dirpath, name))

        # Return the data generators for the training and validation sets
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    'img_paths': img_paths,
                    'mask_paths': mask_paths,
                    'anno_paths': anno_paths
                },
            )
        ]

    def _generate_examples(self, path):
        """Yields examples."""
        # TODO(my_dataset): Yields (key, example) tuples from the dataset
        for f in path.glob('*.jpeg'):
            yield 'key', {
                'image': f,
                'label': 'yes',
            }
