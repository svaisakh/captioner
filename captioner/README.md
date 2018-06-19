## Prerequisites

Before running any script, do the following:

1. Activate the Conda environment by running `source activate captioner`.

2. Export the repository's path to the PYTHONPATH environment variable as follows:

   `export PYTHONPATH=$PYTHONPATH:<path>`

   where `path` is the path where you've cloned this repo.



## For Training

1. Download and extract the COCO dataset and captions using the [download](download.py) script.

   The files are stored in a hidden .data directory in your _home_ folder.

2. Extract the features using a pre-trained ResNet by running the [extract](extract.py) script.

3. Run the [train](train.py) script to train the model.



## For Sampling

Run the [eval](eval.py) script to see the model in action.

Just pass in the filename of the file you wish to caption.

For example, `python eval.py kitten.jpg`



## Customizing Hyperparameters

For each script, you can pass in certain parameters (or hyperparameters).

For example, to run the training script for 10 epochs, go:

`python train.py --epochs=10`

If you want to change the defaults, just edit `hparams.py`



## Getting Help

The `--help` flag will help you out for any script.

For example, `python train.py --help`.