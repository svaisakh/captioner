import magnet as mag
import torch

from captioner.hparams import beam_size, image_shape, vocab_size, hidden_size, num_layers, rnn_type, architecture, probabilistic
from captioner.eval import caption
from captioner.nlp import get_nlp
from captioner.extract import Extractor
from captioner.model import Model

from pathlib import Path
from PIL import Image

def _prepare():
	DIR_CHECKPOINTS = Path(__file__).resolve().parents[1] / 'checkpoints'

	nlp = get_nlp('en_core_web_lg', vocab_size, DIR_CHECKPOINTS / 'vocab')
	embedding_dim = nlp.vocab.vectors.shape[1]

	extractor = Extractor(architecture)
	feature_dim = extractor.feature_size

	model = Model(feature_dim, embedding_dim, hidden_size,
              num_layers, rnn_type, vocab_size)

	model.load_state_dict(torch.load(DIR_CHECKPOINTS / 'model.pt', map_location=mag.device))

	return nlp, extractor, model

nlp, extractor, model = _prepare()

def get_captions(path):
	"""
	Gets a list of (caption, probability) tuples applied on an image at the specified path.
	:param path: Path to the image.
	:return: List of (caption, probability) tuples.
	"""
	image = Image.open(path)
	return caption(image, model, extractor, nlp, beam_size, probabilistic, image_shape)