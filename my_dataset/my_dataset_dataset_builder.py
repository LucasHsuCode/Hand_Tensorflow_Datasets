"""my_dataset dataset."""

import tensorflow_datasets as tfds


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

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(my_dataset): Downloads the data and defines the splits
    path = dl_manager.download_and_extract('https://todo-data-url')

    # TODO(my_dataset): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path / 'train_imgs'),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(my_dataset): Yields (key, example) tuples from the dataset
    for f in path.glob('*.jpeg'):
      yield 'key', {
          'image': f,
          'label': 'yes',
      }
