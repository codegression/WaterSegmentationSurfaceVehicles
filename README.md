
# Water Segmentation for Surface Vehicles Using Deep Learning

## About

This system uses a variation of ResNet with transfer learning to perform water segmentation. It is a work in progress.

## Datasets

Clone the project and create a folder called "datasets" inside the project folder.
Download the following datasets into sub-folders

### Tempere-WaterSeg dataset
[https://etsin.fairdata.fi/dataset/e0c6ef65-6e1e-4739-abe3-0455697df5ab#0](https://etsin.fairdata.fi/dataset/e0c6ef65-6e1e-4739-abe3-0455697df5ab#0)

### IntCatch Dataset
[http://profs.scienze.univr.it/farinelli/intcatchvisiondb/intcatchvisiondb.html](http://profs.scienze.univr.it/farinelli/intcatchvisiondb/intcatchvisiondb.html)
 
### Water segmentation dataset
[https://www.kaggle.com/datasets/gvclsu/water-segmentation-dataset](https://www.kaggle.com/datasets/gvclsu/water-segmentation-dataset)

If you use your own dataset, create a function to load the dataset in data.py


## Training

Run training.py to perform training. Make sure that datasets' paths are correct.
The trainer module will save the weights of the model as "model-weights.h5".


## Modifying architecture
You can modify the architecture by modifying "architecture.py"


## Web demo
Run app.py to run in a debugging environment and server.py to run in a production environment.


## Docker
A docker file has been provided. It can be run inside a docker container.
