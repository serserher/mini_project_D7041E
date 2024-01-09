 # Mini project D7041E: 

# Group 33, see the members below:

- Sergio Serrano Hernández (serser-1): serser-1@gmail.com
- Luna Alvarez Redondo (lunlva-3): lunlva-3@student.ltu.se 
- Alejandro Carrillo Cerdá (alecar-3): alecar-3@student.ltu.se 

For our project, we have decided we are going to compare the performance of two different object detection algorithm approaches. For the first one, we are going to create our own network using PyTorch, while for the second one, will use an already existing out-of-the-box architecture: YoloV3. Along the project, we are also going to use two datasets as we strive to get the maximum grade, i.e: 5 - Using two datasets and two different algorithms.

The file "object_detection_pytorch.ipynb", contains the code for the network created with Pytorch, while the file "object_detection_yolo.ipynb" contains the implementation of YoloV3 approach.

The Dataset is gotten from: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit (click the very first link that says training/validation data).

We are in fact using two Datasets, the full version of the downloaded one, and a reduced version that only includes a certain kind of samples with only one object per sample to be detected.

In order to replicate the results, just clone our repository, download the Dataset and place the folder "VOC2012" in the same folder as the cloned repo (same level as the scripts, so that the directories are reachable without having to change anything in the code), and perform the reduction of the Dataset with the script for reducing the original Dataset.

