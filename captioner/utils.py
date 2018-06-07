import numpy as np

from contextlib import contextmanager
from collections import namedtuple
from itertools import chain

class BeamSearch:
	Branch = namedtuple('Branch', ['content', 'score', 'context'])

	def __init__(self, build=None):
		self.build = build

	def __call__(self, beam_size, context, max_len, probabilistic=0):
		return self._search(beam_size, context, max_len, probabilistic)

	def _search(self, beam_size, context, max_len, probabilistic):
		branches = [self.Branch([], 0, context)]

		for _ in range(max_len):
			branches = list(chain(*[[new_branch
									 for new_branch in self._get_branches(branch, beam_size, probabilistic)]
									for branch in branches]))

			branches = self._prune_branches(branches, beam_size, probabilistic)

		branches = [self.Branch(branch.content, np.exp(branch.score), None) for branch in branches]

		return branches

	def _get_branches(self, branch, beam_size, probabilistic):
		contents, scores, context = self.build(branch.content, branch.context)
		nodes = [self.Branch([content], score, context)
				 for content, score in zip(contents, scores)]

		if not probabilistic: nodes = self._prune_branches(nodes, beam_size, probabilistic)

		return [self._merge(branch, node) for node in nodes]

	def _merge(self, b1, b2):
		return self.Branch(b1.content + b2.content, b1.score + b2.score, b2.context)

	def _prune_branches(self, branches, beam_size, probabilistic):
		branches = _sort_list(branches, key=lambda branch: np.exp(branch.score), probabilistic=probabilistic)
		return branches[:beam_size]

def _sort_list(x, key, probabilistic):
	if not probabilistic:
		x.sort(key=key, reverse=True)
		return x

	probs = np.array([key(x_i) for x_i in x])
	probs /= probs.sum()

	ids = np.random.choice(list(range(len(x))), len(x), replace=False, p=probs)
	return [x[i] for i in ids]

def show_coco(img, captions):
	import matplotlib.pyplot as plt

	from numpy.random import randint

	captions = captions[randint(len(captions))]

	def show_image(image, title):
		image = image.permute(1, 2, 0).numpy().copy()
		i_min, i_max = image.min(), image.max()
		image = (image - i_min) / (i_max - i_min)
		plt.imshow(image)
		plt.xticks([]); plt.yticks([]); plt.grid(False)
		plt.title(title)
		plt.show()

	for i, c in zip(img, captions): show_image(i, c)

def loopy(gen):
	while True:
		for x in iter(gen): yield x

def working_directory(path):
	"""path can also be a function in case of decorator"""

	from inspect import isfunction
	if not isfunction(path):
		return _working_directory_context_manager(path)

	from functools import wraps

	@wraps(path)
	def new_fn(*args, **kwargs):
		from pathlib import PosixPath

		working_path = [a for a in args if type(a) is PosixPath]
		if len(working_path) != 0: working_path = working_path[0]
		else:
			working_path = [v for v in kwargs.values() if type(v) is PosixPath]
			if len(working_path) != 0: working_path = working_path[0]
			else: raise RuntimeError('No suitable paths found')

		with _working_directory_context_manager(working_path):
			return path(*args, **kwargs)

	return new_fn

@contextmanager
def _working_directory_context_manager(path):
	import os

	# Change to working directory
	path_cwd = os.getcwd()
	os.chdir(path)

	yield

	os.chdir(path_cwd) # Change back to working directory

def get_tqdm():
	import tqdm

	try:
		get_ipython
		return getattr(tqdm, 'tqdm_notebook')
	except:
		return getattr(tqdm, 'tqdm')