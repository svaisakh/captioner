## Prerequisites

Follow the general steps in [this](https://github.com/svaisakh/using-lessons) tutorial to set up the environment and the startup file.

**Make sure you have MagNet installed.**



## Getting the server up and running

1. Activate the Conda environment by running `source activate captioner`.

2. Export the repository's path to the PYTHONPATH environment variable as follows:

   `export PYTHONPATH=$PYTHONPATH:<path>`

   where `path` is the path where you've cloned this repo.

3. Navigate to this directory:

   `cd <path_to_this_repo>/serve`

4. Export the FLASK_APP environment variable:

   `export FLASK_APP=main.py`

5. Run the server:

   `flask run --host=0.0.0.0`



## Using the API

The `/caption` endpoint accepts a POST request with an image.

It returns a list of (caption, probability) tuples.



## The microservice

The main server hosts a small upload form.

The submitted image is captioned, and the result is shown along with the image.



## Customizing hyperparameters

Edit the `hparams.py` file in the captioner directory.