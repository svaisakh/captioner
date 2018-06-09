from multiprocessing import cpu_count as __num_cores

# Extractor
architecture = 'resnet34'
extractor_batch_size = 16
image_shape = 224

# General
num_workers = 8

num_workers = min(num_workers, __num_cores())

# Training
rnn_type = 'lstm'
hidden_size = 512
num_layers = 1
shuffle = True

learning_rate = 1e-4
optimizer = 'adam'

epochs = 1
iterations = None
save_every = 5
write_every = 1

def _get_optimizer(optimizer):
	from torch import optim
	from functools import partial

	_optim_dict = {'adam': partial(optim.Adam, amsgrad=True)}
	return _optim_dict[optimizer]

optimizer = _get_optimizer(optimizer)

# SpaCy
vocab_size = 10000
prune_batch_size = 8
caption_idx = 0

# Evaluation
beam_size = 3
probabilistic = False

def _get_click_options():
	import click

	return \
	{
		'architecture':
		{
			'help': "The ResNet Architecture to use for extraction.",
			'type': click.Choice(('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'))
		},
		'image_shape':
		{
			'help': "The square size that all images will be resized to prior to extraction. Note that Torchvision's ResNet models only support 224x224 images."
					" So this is the default value.",
		},
		'extractor_batch_size':
		{
			'help': "The extractor will process images in batches of this size. Use as big a value as your machine can handle. A power of two is preferred."
		},
		'num_workers':
		{
			'help': f"Number of CPU cores to use for loading images from disk. Maximum that can be used in your machine is {__num_cores()}"
		}
	}

click_options = _get_click_options()