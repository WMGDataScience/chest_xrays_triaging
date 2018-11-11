# chest x-rays triaging

This repository contains the code used for the paper ``Automated triaging of 
adult chest radiographs using deep artificial neural networks``.

It includes scripts for:
* training a Deep Convolutional Network (DCN).
* evaluate the prediction of the DCN using models ensembles.
* simulate the triaging process.


## Dependencies

* Have at least CUDA 7.0  
* cudnn R4 or more  
* Torch7:
	+ base torch packages (included in the torch installation)
	+ tds
	+ rnn
* Python 2.7:
	+ [torchfile library](https://github.com/mauann/python-torchfile)
	+ numpy
	+ pymongo
	+ seaborn (for graphs)
    + matplotlib (for graphs)
    + sklearn


## How to use it?

### Train a new DCN

The input of the script includes a .csv file with the chest x-ray images filepath 
and the list of the associated abnormalites.

Run the command:
```
th doall.lua <options> 
```
take a look at the input options, you need to configure these options in the proper way before running the script.  

### Evaluate a trained DCN (or an ensemble of DCNs)

The input of the script includes the one or more .t7 binary files with 
the predictions of one or more DCNs. Moreover the script needs a MongoDB with
the chest xrays metadata.

Run the command:
```
python evaluate_predictions/cf_from_net_results.py <options>
```
take a look at the input options, you need to configure these options in 
the proper way before running the script.

### Simulate the triaging process

The input of the script includes the one or more .t7 binary files with 
the predictions of one or more DCNs. Moreover the script needs a MongoDB with
the chest xrays metadata.

Run the command:
```
python reporting_delays_simulation/simulate_reporting.py <options>
```
take a look at the input options, you need to configure these options in the proper way before running the script.


## DCN implemented models

All the implemented models are in the ``models/`` directory.

* VGG like net: a simple model that use the principle described [here](http://arxiv.org/pdf/1409.1556.pdf).
* VGG like big net: the same principle used for 1) but with bigger input image size (3x407x407).
* [Inception v3](http://arxiv.org/pdf/1512.00567v3.pdf).
* Inception v3 modified: 1211x1083 input images.
* [Inception v4](http://arxiv.org/pdf/1602.07261v1.pdf) originally implemented [here](https://github.com/itaicaspi/inception-v4.torch).
* [Residual networks](https://arxiv.org/pdf/1512.03385v1.pdf) originally implemented [here](https://github.com/facebook/fb.resnet.torch).
* [ResNext](https://arxiv.org/pdf/1611.05431.pdf) originally implemented [here](https://github.com/facebookresearch/ResNeXt).
