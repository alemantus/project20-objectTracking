# project20-objectTracking
This repository contains a notebook based on a pytorch turtorial found here: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
It has been modified to work with data using the PascalVOC format. 

## How to use
The folder data extraction contains all the code nescessary to convert the two .avi files into frames. If the user wants to make one large dataset it is possible using the mergedata.py script. 

The train folder contains the notebook and all of its dependencies required to train the model. 

The demo folder contain a simple script, objectTrackerDemo.py, that uses a trained model to track the desired object. Due to size contraints no model is uploaded here. 

The project_20_data contains all of our data along with frames and annotations
