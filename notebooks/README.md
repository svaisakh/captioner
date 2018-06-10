## Prerequisites

Follow the general steps in [this](https://github.com/svaisakh/using-lessons) tutorial to set up the environment and the startup file.

**Make sure you have MagNet installed.**



## For Training

1. Download and extract the COCO dataset and captions using the [Download](Download.ipynb) notebook.

   The files are stored in a hidden .data directory in your _home_ folder.

2. Extract the features using a pre-trained ResNet by running the [Extract](Extract.ipynb) notebook.
3. Run the [Train](Train.ipynb) notebook to train the model and visualize the losses.



## For Sampling

Run the [Caption](Caption.ipynb) notebook to see the model in action.

You can provide your own images too!



## Customizing Hyperparameters

Inside each notebook you can, of course, change the value of any hyperparameter dynamically.

If you want to change the defaults, just edit `hparams.py`