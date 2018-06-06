import magnet as mag
from train import process_caption

def caption():
	img, caption = next(dl)
	feature = extractor(img.to(mag.device))
	show_images(img.permute(0, 2, 3, 1).numpy(), pixel_range='auto', retain=True)
	caption = sample(feature, caption[0])
	caption = caption[:caption.find('.')]
	plt.title(caption)
	plt.show()

def sample(model, feature, nlp, cap=None):
	from nlp import idx_word

	if cap is not None:
		cap = process_caption(cap[0])[0]
		y = model(feature.to(mag.device), cap.to(mag.device))
	else:
		y = model(feature.to(mag.device), nlp=nlp)

	indices = y.squeeze(0)
	if cap is not None: indices = indices.max(-1)[1]

	caption = ' '.join([idx_word(i.item(), nlp) for i in indices])
	caption = caption[:caption.find('.')]
	return caption