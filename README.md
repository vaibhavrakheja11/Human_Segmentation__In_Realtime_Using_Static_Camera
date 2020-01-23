## Deep Sequence learning for wildfire spread prediction

## Projet Introduction

In North America, wildfires occur frequently and have direct impacts on society, economy, and environment. In Canada, wildfire management agencies spend 800 million dollars each year on wildfire management, which is projected to increase due to the human-introduced climate change. Therefore, improving the efficiency of the fire management decision-making process is one of the most important tasks for wildfire management agencies. In particular, wildfire spread prediction is one of the essential yet challenging tasks. 

Current methods in place for wildfire spread prediction is based either on empirical or physical models, which are all limited by the imperfect understanding of the physical processes and data quality. The recent advancement in pixel-wise deep sequence learning models may be a feasible solution for this problem. However, there are no existing deep learningbased works in the context of wildfire spread prediction, In addition, there is no existing standard (e.g., how to build training dataset and framework architecture) for using deep learning methods for wildfire spread prediction. 

As the first step of a complicated question, we propose to implement the pixel-wise deep learning methods that combine the convolutional–recurrent approaches for wildfire spread prediction. We will build the training dataset using the historical wildfire mapped from the satellite images. For evaluation, we will use our own benchmark dataset following the standard object segmentation evaluation methods.

------

## Getting started

To start trainning with real wildfire data.

Open Jupyter Notebook from terminal (or any other method you preferred)

```cmd
$ jupyter notebook
```

If you get problem as the following: `$ UnicodeDecodeError: 'ascii' codec can't decode byte 0xe5 in position 4: ordinal not in range(128)`, then try:

```python
$ LANG=zn jupyter notebook
```

Then if you want to try model without any additional features, please open file  [tiffworksheet.ipynb](http://localhost:8888/notebooks/tiffworksheet.ipynb) otherwise, the model in [TiffWorksheetAdditionalFeatures-Copy1.ipynb](http://localhost:8888/notebooks/TiffWorksheetAdditionalFeatures-Copy1.ipynb) contain supplementary features.

After open the .ipynb file, run each section in order, you may see some examples and stardard images at the beginning. After running the section start with `# Conv2DLSTM with Gaussian Noise` ( in tiffworksheet.ipynb) or the section start with `#ConvoLstm2d with gaussian noise`( in TiffWorksheetAdditionalFeatures-Copy1.ipynb), the corresponding model start training with real fire data-set which may cost some time.

Also you can cotumize the parameters of model in following section:

```python
#initialise params
hyperparams = {"num_epochs": 10, 
          "batch_size": 2,
          "height": 128,
          "width": 128}

config=hyperparams
```

After trainning, the last block in the file (starting with `#local testing block uttu` ) will give you the test images and evaluation results. There will be two columns of results, ground-truth is on the left and the right side is predictions
