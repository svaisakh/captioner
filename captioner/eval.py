import magnet as mag

from PIL import Image
from pathlib import Path

from captioner.data import get_transform

@mag.eval
def caption(model, extractor, nlp, image, beam_size=1, image_shape=None):
	transform = get_transform(image_shape)
	if type(image) in ('str', Path): image = Image.open(filenames[0])

	features = extractor(transform(image).unsqueeze(0).to(mag.device))

	captions = model(features, nlp=nlp, beam_size=beam_size)
	if len(captions) == 1: return captions[0][0]

	return '\n'.join(f'{c} ({p:.2f})' for c, p in captions)