from multiprocessing import cpu_count as __num_cores

# Extractor
architecture = 'resnet34'
image_shape = 224
extractor_batch_size = 16
num_workers = 8

num_workers = min(num_workers, __num_cores())

# Captions
vocab_size = 10000
caption_idx = 0
prune_batch_size = 8

# Model
rnn_type = 'LSTM'
hidden_size = 512
num_layers = 1

# Training
epochs = 10.0
iterations = -1
shuffle = True
optimizer = 'adam'
learning_rate = 1e-4
save_every = 5.0
write_every = 1.0

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
		},
		'vocab_size':
		{
			'help': "The size of the vocabulary to be used. The larger the size, the more rare words will be covered. This should not be as much of an issue since"
					" SpaCy automatically replaces unknown words with their synonyms at runtime. Beware though, that a larger size will consume tons of memory since"
					" we'll be doing a softmax on the scores."
		},
		'caption_idx':
		{
			'help': "The index of the captions to be used for training. The COCO dataset has 5 captions per image. While training, one of these is chosen according to"
					" the value of this parameter. If -1, indices will be randomly chosen at runtime for each image.",
			'type': click.IntRange(-1, 5)
		},
		'prune_batch_size':
		{
			'help': "SpaCy's vocabulary is pruned to vocab_size in batches of this size."
		},
		'rnn_type':
		{
			'help': "The type of RNN to use. Karpathy et.al. report better performance with an LSTM.",
			'type': click.Choice(('RNN', 'LSTM'))
		},
		'hidden_size':
		{
			'help': "The hidden size of RNN layers."
		},
		'num_layers':
		{
			'help': "Number of stacked RNN layers to use."
		},
		'epochs':
		{
			'help': "Number of epochs to train."
		},
		'iterations':
		{
			'help': "Number of iterations to train. If this is positive, then epochs is overriden with this. Useful for debugging (eg. train for 1 iteration)."
		},
		'shuffle':
		{
			'help': "Whether to shuffle the dataset while training. Shuffling generally produces better performance since the model can't overfit to specific batches."
		},
		'optimizer':
		{
			'help': "The type of optimizer to use. When the papers came out, they reported using RMSProp. Adam was invented shortly afterwards. Use it (with AMSGrad)!",
			'type': click.Choice(('adam', ))
		},
		'learning_rate':
		{
			'help': "Learning rate. Constant throughout training."
		},
		'save_every':
		{
			'help': "The frequency (number of minutes) with which the model is saved during training."
		},
		'write_every':
		{
			'help': "The frequency (number of minutes) with which the training history is appended."
		},
		'beam_size':
		{
			'help': "Captions are sampled using beam search with this size."
		},
		'probabilistic':
		{
			'help': "If True, the beam search retains nodes at each iteration according to their probabilities."
		}
	}

click_options = _get_click_options()