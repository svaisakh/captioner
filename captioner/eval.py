import magnet as mag

from PIL import Image
from pathlib import Path

from captioner.data import get_transform

@mag.eval
def caption(image, model, extractor, nlp, beam_size=1, probabilistic=False, image_shape=None):
	"""
	Captions a PIL image.

	:param image: The PIL image which needs to be captioned.
	:param model: The trained RNN generative model.
	:param extractor: The pretrained CNN which was used for extraction.
	:param nlp: spaCy model which will be used for tokenization. Take care to use the same model and vocabulary size
				which was used while training since the vectors will differ otherwise.
	:param beam_size: Captions are sampled using beam search with this size.
	:param probabilistic: If True, the beam search retains nodes at each iteration according to their probabilities.
	:param image_shape: The shape that all images will be resized to prior to extraction. If an integer is provided, it applies to both the dimensions.
	:return: A list of (caption, probability) tuples.
	"""
	transform = get_transform(image_shape)
	if type(image) in ('str', Path): image = Image.open(filenames[0])

	features = extractor(transform(image).unsqueeze(0).to(mag.device))

	return model(features, nlp=nlp, beam_size=beam_size, probabilistic=probabilistic)

def pretty_print(captions):
	"""
	Nicely formats a list of (caption, probability) tuples for human consumption.

	:param captions: The list of captions and corresponding probabilities.
	:return: An aesthetic string.
	"""
	if len(captions) == 1: print(captions[0][0]); return

	print('\n'.join(f'{c} ({p:.2f})' for c, p in captions))



def __main(beam_size, probabilistic, image_shape, hidden_size, num_layers, rnn_type, vocab_size, architecture):
	from captioner.nlp import get_nlp
	from captioner.extract import Extractor
	from captioner.model import Model

	device = 'cuda:0' if mag.device == 'cuda' else mag.device
	nlp = get_nlp('en_core_web_lg', vocab_size, DIR_CHECKPOINTS / 'vocab')
	embedding_dim = nlp.vocab.vectors.shape[1]

	extractor = Extractor(architecture)
	feature_dim = extractor.feature_size

	model = Model(feature_dim, embedding_dim, hidden_size,
              num_layers, rnn_type, vocab_size)
	model.load_state_dict(torch.load(DIR_CHECKPOINTS / 'model.pt', map_location=device))

	print('\nCaption:\n')
	pretty_print(caption(img, model, extractor, nlp, beam_size, probabilistic, image_shape))

if __name__ == '__main__':
	from captioner import hparams
	from captioner.utils import launch

	launch(__main, default_module=hparams)