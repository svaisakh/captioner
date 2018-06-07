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