# Facial-Expression-Recognition
** Note that this project is just for reference purposes. Some file paths need to be fixed on respective computers. So please check the console for any missing file path errors. **

** Make sure to install all the required packages mentioned in prerequisites section under project requirements. **

** I used quite a number of links to set up environment on my machine. Be sure to google and set tensorflow(GPU) properly on your work station. **

** I hope that you are familiar with most of the commands to install and manage packages in anaconda **

** After installing Keras, make sure it is using tensorflow as backend **

## Abstract
In this project, Facial Expression Recognition system is implemented using CNN techniques. The software is based on extraction of facial features using suitable algorithms of the face geometry and approximation techniques such as Convolution two dimensional and three dimensional algorithms. The experiments employed to evaluate
our technique were carried out using kaggle dataset (fer2013.csv). The proposed method achieves competitive results when compared with other facial expression recognition methods – 91% of accuracy. It is fast to train, and it allows for real time
facial expression recognition with standard computers.

## Project Requirements
- Software: Anaconda Navigator(Python), Spyder, Flask, CSS3, HTML5, Javascript
- Operating System: Windows/Mac/Linux
- Hardware: 16GB RAM, Nvidia GTX 1070 (8GB Video Memory)
- Prerequisites: Tensorflow (GPU), Keras, OpenCV, h5py, Pandas, Numpy, Scikit-Learn

## Installation
On windows machine, install Anaconda which is a package manager and provides python virtual environment. Once installed either use inbuilt package manager or use "conda" / "pip" to install all the packages in respective environment. Following the equivalent steps in Ubuntu shoudl yield the same results.

## Usage - Individual files(You can use these files to see how each module works)
Once everything is set up, you can go to "Prediction_Video" folder and run "Prediction_Video.py" file. Likewise, You could also go to "Model_Prediction" folder and run "prediction.py" file.

## Usage - Flask application(You can use the bundled flask application to see how the application works on a web server)
Individual files have been combined to create a application using python flask. This application can run on web servers. Additional code of **Speech Generation** and **Realtime Webcam Detection** has been added to this application.
