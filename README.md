# Project - Classification of images extracted from histopathologic scans of lymph node sections

### Presentation of results - for teachers

This project was developed as a part of course CM2003 at KTH. The final report and the power point presentation are found as pdf files in the Documents folder. The code is found in the src folder. 

### Task

This is binary classification task with a dataset that are already class weighted. All python files in the root directory are main files. This pipeline uses configuration files for training and prediction, and the files can be found in the InputFiles folder. Keras implementations of VGG, ResNet, and DenseNet can be found in the Models folder. Utility functions, e.g. to read in json files and plot resluts are found in the Utils folder. The Results folder and its subfolders are created automatically within the scripts.   

The user can view the dataset by stepping through 64 images at a time using the show_dataset file, and can generate plots in arbitrary ways using csv history files from past training round, using plot_history. These two main files in the root folder uses argparsers. 

Commands to run different main files can be found in the commands.txt file, and also in each main file. The full filesystem with Code as root can be seen in the tree diagram below. The data should be put in /Code/dl_data/Camelyon/, which is the case in the tree diagram. The files can be downloaded from https://patchcamelyon.grand-challenge.org/Introduction/. 

<img width=280 alt="file_system" src="https://user-images.githubusercontent.com/55019110/66723566-cac55080-ee1a-11e9-92b2-33c8cae2556b.png">


