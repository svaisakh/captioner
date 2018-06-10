import numpy as np

from contextlib import contextmanager
from collections import namedtuple
from itertools import chain

def _get_data_paths():
	from pathlib import Path

	DIR_DATA = Path('~/.data/COCO').expanduser()
	DIR_CHECKPOINTS = Path(__file__).resolve().parents[1] / 'checkpoints'

	for directory in [DIR_DATA, DIR_CHECKPOINTS]: directory.mkdir(exist_ok=True, parents=True)
	return DIR_DATA, DIR_CHECKPOINTS

DIR_DATA, DIR_CHECKPOINTS = _get_data_paths()

class BeamSearch:
	Branch = namedtuple('Branch', ['content', 'score', 'context'])

	def __init__(self, build=None):
		"""
		A Beam Searcher instance.

		:param build: A function, f(content, context) which takes in a content and context object and
					returns a (contents, scores, context) tuple.
					The definition of what these are is upto the function.

					Content represents the things in a particular node.
					Context represents the features of a branch of nodes.
					A score is the log probability of a particular node.
		"""
		self.build = build

	def __call__(self, beam_size, context, max_len, probabilistic=0):
		"""
		Perform beam search.

		:param beam_size: The size of the beam used.
		:param context: The initial context.
		:param max_len: Search will be terminated when branches reach this length.
		:param probabilistic: If True, retains nodes at each iteration according to their probabilities.
		:return: A list of branches found. Each branch has a content and score (relative probability).
		"""
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

	@staticmethod
	def _prune_branches(branches, beam_size, probabilistic):
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

def launch(fn, defaults=None, default_module=None):
	"""
	Launches a function as the main entry point to a command line program.

	:param fn: The function to launch.
	:param defaults: A dictionary of default arguments to the function.
	:param default_module: A module with default arguments with the same name.
						Any options to click should be given in a dictionary called click_options in this module.
	"""
	import click

	from inspect import signature

	args = list(signature(fn).parameters.keys())

	click_options = getattr(default_module, 'click_options', {}) if default_module is not None else {}

	for k in args[::-1]:
		if defaults is not None and k in defaults.keys():
			d = defaults[k]
			if type(d) in (tuple, list):
				fn = click.option('--' + k, show_default=True, default=d[0], help=d[1])(fn)
			else:
				fn = click.option('--' + k, show_default=True, default=d)(fn)

		if hasattr(default_module, k):
			kwargs = click_options[k] if k in click_options.keys() else {}
			fn = click.option('--' + k, show_default=True, default=getattr(default_module, k), **kwargs)(fn)

	fn = click.command()(fn)

	return fn()

def show_coco(img, captions):
	"""
	Show the coco images with the captions as title.
	:param img: Images to show.
	:param captions: Corresponding captions
	"""
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
	"""
	Returns an iterator with infinite length.
	Does not raise the StopException.

	:param gen: The generator object to loop.
	:return: An infinite iterator.
	"""
	while True:
		for x in iter(gen): yield x

def working_directory(path):
	"""
	A context manager cum decorator which changes the working directory to the specified path.
	If used as a decorator with no arguments, the first path in the arguments of the inner function is used.

	:param path: The path to change to/the function to decorate
	"""
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
	"""
	:return: Returns a flexible tqdm object according to the environment of execution.
	"""
	import tqdm

	try:
		get_ipython()
		return getattr(tqdm, 'tqdm_notebook')
	except:
		return getattr(tqdm, 'tqdm')

def get_optimizer(optimizer):
	"""
	Returns an optimizer according to the passed string.
	:param optimizer: The string representation of the optimizer. eg. 'adam' for Adam etc.
	:return: The proper nn.optim optimizer.
	"""
	from torch import optim
	from functools import partial

	_optim_dict = {'adam': partial(optim.Adam, amsgrad=True)}
	return _optim_dict[optimizer]