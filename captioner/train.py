import magnet as mag

from torch.nn import functional as F

from captioner.nlp import process_caption

def optimize(model, optimizer, history, dataloader, nlp, save_path, epochs=1, iterations=-1, save_every=5,
			 write_every=1):
	"""
	Trains the model for the specified number of epochs/iterations.

	This method also handles checkpointing of the model and optimizer.

	:param model: The RNN generative model to train.
	:param optimizer: The optimizer to use for training.
	:param history: A Training history dictionary with the following keys: ['iterations', 'loss', 'val_loss'].
					It will be updated during training with the statistics.
	:param dataloader: A dictionary containing the training and validation DataLoaders with keys 'train' and 'val' respectively.
	:param nlp: The spaCy model to use for training.
	:param save_path: The model and optimizer state will be periodically saved to this path.
	:param epochs: The number of epochs to train.
	:param iterations: Number of iterations to train. If this is positive, then epochs is overriden with this. Useful for debugging (eg. train for 1 iteration).
	:param save_every: The frequency (number of minutes) with which the model is saved during training.
	:param write_every: The frequency (number of minutes) with which the training history is appended.
	"""
	import torch

	from captioner.utils import get_tqdm, loopy
	from time import time

	tqdm = get_tqdm()
	start_time_write = time()
	start_time_save = time()
	mean = lambda x: sum(x) / len(x)

	model.train()
	if iterations < 0: iterations = int(epochs * len(dataloader['train']))
	prog_bar = tqdm(range(iterations))
	gen = {mode: loopy(dataloader[mode]) for mode in ('train', 'val')}
	running_history = {'loss': []}

	device = 'cuda:0' if mag.device == 'cuda' else mag.device

	for batch in prog_bar:
		feature, caption = next(gen['train'])
		loss = _get_loss(model, feature, caption[0], nlp)

		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		running_history['loss'].append(loss.item())
		history['iterations'] += 1

		if (time() - start_time_write > write_every * 60) or (batch == iterations - 1):
			start_time_write = time()
			mean_loss = mean(running_history['loss'])
			history['loss'].append(mean_loss)
			running_history['loss'] = []

			feature, caption = next(gen['val'])
			with mag.eval(model): loss = _get_loss(model, feature, caption[0], nlp).item()
			history['val_loss'].append(loss)

			prog_bar.set_description(f'{mean_loss:.2f} val={loss:.2f}')

		if (time() - start_time_save > save_every * 60) or (batch == iterations - 1):
			start_time_save = time()
			torch.save(model.state_dict(), save_path / 'model.pt')
			torch.save(optimizer.state_dict(), save_path  / 'optimizer.pt')

def _get_loss(model, feature, caption, nlp):
	cap, target = process_caption(caption, nlp)
	y = model(feature.to(mag.device), cap.to(mag.device))
	return F.cross_entropy(y.squeeze(0), target.to(mag.device))

def __main(epochs, iterations, shuffle, optimizer, learning_rate, vocab_size, caption_idx, hidden_size, num_layers, rnn_type):
	from captioner.data import get_training_dataloaders
	from captioner.nlp import get_nlp
	from captioner.utils import DIR_DATA, DIR_CHECKPOINTS, get_optimizer

	if not (DIR_DATA / 'train' / 'features.pt').exists():
		print("Features don't seem to be extracted or cannot be found. Run extract.py once again, maybe?")
		return

	device = 'cuda:0' if mag.device == 'cuda' else mag.device

	print('Loading SpaCy into memory with', vocab_size, 'words.')
	nlp = get_nlp('en_core_web_lg', vocab_size, DIR_CHECKPOINTS / 'vocab')
	embedding_dim = nlp.vocab.vectors.shape[1]

	print('Getting data.')
	caption_idx = None if caption_idx == 'None' else int(caption_idx)
	dataloader = get_training_dataloaders(DIR_DATA, caption_idx, shuffle)
	x = next(iter(dataloader['val']))
	feature_dim = x[0].shape[1]

	print('Creating the model with:\nfeature_dim =', feature_dim, '\nembedding_dim =', embedding_dim, '\nhidden_size = ',
		  hidden_size, '\nnum_layers = ', num_layers, '\nrnn_type = ', rnn_type)

	model = Model(feature_dim, embedding_dim, hidden_size,
			  num_layers, rnn_type, vocab_size)

	if (DIR_CHECKPOINTS / 'model.pt').exists(): model.load_state_dict(torch.load(DIR_CHECKPOINTS / 'model.pt', map_location=device))

	print('Using the {optimizer} optimizer.')

	if isinstance(optimizer, str): optimizer = get_optimizer(optimizer)
	optimizer = optimizer(model.parameters(), learning_rate)
	if (DIR_CHECKPOINTS / 'optimizer.pt').exists():
		optimizer.load_state_dict(torch.load(DIR_CHECKPOINTS / 'optimizer.pt', map_location=device))

	history = {'iterations': 0, 'loss': [], 'val_loss': []}

	print(f"Training for {iterations/len(dataloader['train']):.2f} epochs ({iterations} iterations)")
	print('Will save the model to disk every', save_every, 'minutes.')
	print('\n\t\t\tHere we go!')

	optimize(model, optimizer, history, dataloader, nlp, DIR_CHECKPOINTS, epochs, iterations, save_every)

	print('Done.')

if __name__ == '__main__':
	import hparams

	from captioner.utils import launch

	launch(__main, default_module=hparams)