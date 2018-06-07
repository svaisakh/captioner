import magnet as mag

from PIL import Image
from pathlib import Path

from captioner.data import get_transform

@mag.eval
def caption(model, extractor, nlp, image, beam_size=1, image_shape=None):
	transform = get_transform(image_shape)
	if type(image) in ('str', Path): image = Image.open(filenames[0])

	features = extractor(transform(image).unsqueeze(0).to(mag.device))

	return model(features, nlp=nlp, beam_size=beam_size)

def pretty_print(captions):
	if len(captions) == 1: print(captions[0][0])

	print('\n'.join(f'{c} ({p:.2f})' for c, p in captions))