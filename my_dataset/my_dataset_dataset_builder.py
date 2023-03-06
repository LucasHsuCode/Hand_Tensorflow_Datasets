"""my_dataset dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
from typing import List
import numpy as np
import argparse
import json
import os


class Builder(tfds.core.GeneratorBasedBuilder):
    """
    Custom dataset containing hand images, masks, and annotations downloaded from a remote server.

    The dataset contains images, masks, and annotations for various hand poses and gestures, and is intended for use in
    machine learning tasks such as hand pose estimation and gesture recognition.

    """

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """
        Returns the dataset metadata.

        Returns:
            A `tfds.core.DatasetInfo` object describing the dataset.
        """

        # Specifies the tfds.core.DatasetInfo object
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

    def _split_generators(self, dl_manager: tfds.download.DownloadManager, **kwargs) -> List[tfds.core.SplitGenerator]:

        """Splits the dataset into training and validation sets.

        Args:
            dl_manager (tfds.download.DownloadManager): The download manager.

        Returns:
            A list of `tfds.core.SplitGenerator` objects for generating the training and validation sets.
        """

        # Set the remote server name, username, password, and file paths
        server_name = os.environ['TFDS_SERVER_NAME']
        user_name = os.environ['TFDS_USER_NAME']
        password = os.environ['TFDS_PASSWORD']

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

    def _generate_examples(self, img_paths: List[str], mask_paths: List[str], anno_paths: List[str]):
        """
        Generates examples for the dataset.

        Args:
            img_paths (List[str]): A list of file paths for the images.
            mask_paths (List[str]): A list of file paths for the segmentation masks.
            anno_paths (List[str]): A list of file paths for the annotations.

        Yields:
            Tuple[str, Dict[str, Union[str, np.ndarray]]]: A tuple containing the unique identifier for the example and
            a dictionary containing the image and segmentation mask data.
        """

        # Create lists for annotation file paths
        _anno_paths = [path for path in anno_paths if os.path.basename(path).startswith('annotation')]
        img_anno_paths = [path for path in anno_paths if os.path.basename(path).startswith('images')]
        mask_anno_paths = [path for path in anno_paths if os.path.basename(path).startswith('mask')]

        # Create a mapping from image IDs to file paths
        image_id_to_path = {}
        for image_path in img_anno_paths:
            with open(image_path, 'r') as f:
                image_data = json.load(f)
            for image in image_data['images']:
                image_id = image['id']
                image_path = image['image_path']
                image_id_to_path[image_id] = image_path

        # Create a mapping from mask IDs to file paths
        mask_id_to_path = {}
        for mask_path in mask_anno_paths:
            with open(mask_path, 'r') as f:
                mask_data = json.load(f)
            for mask in mask_data['mask']:
                mask_id = mask['id']
                mask_path = mask['mask_path']
                mask_id_to_path[mask_id] = mask_path

        # Traverse all annotation file paths
        for anno_path in _anno_paths:
            # Read JSON file
            with open(anno_path, 'r') as f:
                anno_data = json.load(f)

            # Traverse all annotations
            for anno in anno_data['annotations']:
                image_id = anno['image_id']
                mask_id = anno['mask_id']

                # Only yield examples if both the image and mask exist
                if image_id in image_id_to_path and mask_id in mask_id_to_path:
                    image_path = image_id_to_path[image_id]
                    mask_path = mask_id_to_path[mask_id]

                    # Map the image path and mask path to their respective lists
                    for path in img_paths:
                        if image_path in path:
                            image_path = path
                    for path in mask_paths:
                        if mask_path in path:
                            # Load the mask and expand the dimensions to match the shape of the image
                            mask = np.load(path)
                            mask = mask.astype(np.uint8)
                            mask = np.expand_dims(mask, axis=-1)
                            example = {
                                'image': image_path,
                                'segmentation': mask
                            }
                            yield image_id, example
                else:
                    continue
