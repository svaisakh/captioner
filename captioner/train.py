import magnet as mag

from torch.nn import functional as F

from captioner.nlp import process_caption

def optimize(model, optimizer, history, dataloader, nlp, vocab_size, save_path, epochs=1, iterations=None, save_every=5, write_every=1):
	import torch

	from captioner.utils import get_tqdm, loopy
	from time import time

	tqdm = get_tqdm()
	start_time_write = time()
	start_time_save = time()
	mean = lambda x: sum(x) / len(x)

	model.train()
	if iterations is None: iterations = int(epochs * len(dataloader['train']))
	prog_bar = tqdm(range(iterations))
	gen = {mode: loopy(dataloader[mode]) for mode in ('train', 'val')}
	running_history = {'loss': []}

	for batch in prog_bar:
		feature, caption = next(gen['train'])
		loss = get_loss(model, feature, caption[0], nlp, vocab_size)

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
			with mag.eval(model): loss = get_loss(model, feature, caption[0], nlp, vocab_size).item()
			history['val_loss'].append(loss)

			prog_bar.set_description(f'{mean_loss:.2f} val={loss:.2f}')

		if (time() - start_time_save > save_every * 60) or (batch == iterations - 1):
			start_time_save = time()
			torch.save(model.state_dict(), save_path / 'model.pt')
			torch.save(optimizer.state_dict(), save_path  / 'optimizer.pt')

def get_loss(model, feature, caption, nlp, vocab_size):
	cap, target = process_caption(caption, nlp, vocab_size)
	y = model(feature.to(mag.device), cap.to(mag.device))
	return F.cross_entropy(y.squeeze(0), target.to(mag.device))