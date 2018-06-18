import Algorithmia, os

from base64 import b64encode

API_KEY = os.environ.get('API_KEY') or ''
client = Algorithmia.client(API_KEY)

algo_url = os.environ.get('ALGO_URL') or ''
algo = client.algo(algo_url)

def get_captions(image):
	"""
	Gets a list of (caption, probability) tuples applied on the image.
	:param image: The image that needs to be captioned or it's path
	:return: List of (caption, probability) tuples.
	"""
	if isinstance(image, str):
		with open(image, 'rb') as f: image = b64encode(f.read())
	else:
		image = b64encode(image.read())
	image = image.decode('utf-8')
	input_ = {'image': image}

	return algo.pipe(input_).result