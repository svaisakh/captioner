import magnet as mag

from PIL import Image
from pathlib import Path

from captioner.data import get_transform

@mag.eval
def caption(model, extractor, nlp, image, beam_size=1, probabilistic=False, image_shape=None):
	transform = get_transform(image_shape)
	if type(image) in ('str', Path): image = Image.open(filenames[0])

	features = extractor(transform(image).unsqueeze(0).to(mag.device))

	return model(features, nlp=nlp, beam_size=beam_size, probabilistic=probabilistic)

def pretty_print(captions):
	if len(captions) == 1: print(captions[0][0]); return

	print('\n'.join(f'{c} ({p:.2f})' for c, p in captions))



def __main(beam_size, probabilistic, image_shape, hidden_size, num_layers, rnn_type, vocab_size, architecture):
	device = 'cuda:0' if mag.device == 'cuda' else mag.device
	nlp = get_nlp('en_core_web_lg', vocab_size, DIR_CHECKPOINTS / 'vocab')
	embedding_dim = nlp.vocab.vectors.shape[1]

	extractor = Extractor(architecture)
	feature_dim = extractor.feature_size

	model = Model(feature_dim, embedding_dim, hidden_size,
              num_layers, rnn_type, vocab_size)
	model.load_state_dict(torch.load(DIR_CHECKPOINTS / 'model.pt', map_location=device))

	print('\nCaption:\n')
	pretty_print(caption(model, extractor, nlp, img, beam_size, probabilistic, image_shape))

if __name__ == '__main__':
	from captioner import hparams
	from captioner.utils import launch

	launch(__main, default_module=hparams)