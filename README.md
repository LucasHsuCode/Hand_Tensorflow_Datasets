# Cri Hand Datasets
This is the official repository for CRI Hand Datasets, a new dataset for computer vision research.
Using Tensorflow Dataset API to build hand datasets

## Dataset Description
CRI Hand Datasets is a dataset of images with segmentation masks. Each image has a resolution of 400x640 pixels, and each segmentation mask has the same resolution but with only one channel. The dataset is designed for use in computer vision research, especially in the field of image segmentation.

## Citation
``` 
@article{my_dataset,
  title={CRI CoreHand Datasets},
  author={Lucas, Frank},
  journal={Journal of My Dataset},
  year={2023},
}
``` 

## Getting Started

To get started with CRI Hand Datasets, you should access a dataset server provided by CRI, you need to obtain the server IP address, username, and password. Then, execute the following commands in the terminal to set up the necessary environment variables:
``` 
export TFDS_SERVER_NAME=192.###.##.##9
export TFDS_USER_NAME=L######0619
export TFDS_PASSWORD=V#####ru8\ ~
```
Note that the value of the TFDS_PASSWORD variable and the whitespace after it are both meaningful, so do not modify them.

Environment variables are only effective in the current terminal session. If you want to use these variables in other terminals or sessions, you need to set them up again.

If you encounter any problems when reading environment variables, you can check the correctness of the command or try reopening the terminal and entering the command again.



## Setting Up Environment
Here are the steps to set up the environment:

1. Create a new conda environment: conda create --name myenv python=3.8
2. Clone the repository: gh repo clone LucasHsuCode/Hand_Tensorflow_Datasets
3. Navigate to the cloned repository: cd Hand_Tensorflow_Datasets
4. Install the required packages: pip install -r requirements.txt
5. Navigate to the dataset folder: cd my_dataset
6. Build the dataset: tfds build --register_checksums
``` 
conda create --name myenv python=3.8
gh repo clone LucasHsuCode/Hand_Tensorflow_Datasets
cd Hand_Tensorflow_Datasets
pip install -r requirements.txt
cd my_dataset
tfds build --register_checksums  # This will generate the dataset files and store them in the ~/tensorflow_datasets directory by default.
```

Then, you can load the dataset using the following code:
``` 
import tensorflow_datasets as tfds
my_dataset = tfds.load('my_dataset', split='train')
``` 
This will load the training split of MyDataset as a TensorFlow Dataset object.

# License
CRI Hand Datasets is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License. See the LICENSE file for more information.