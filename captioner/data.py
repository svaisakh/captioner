from torchvision.datasets import CocoCaptions
from numpy.random import randint

def get_extract_dataloaders(data_path, image_shape=None, batch_size=1, num_workers=0):
	"""
	Makes PyTorch DataLoaders for extraction.

	:param data_path: The root path of the COCO dataset.
	:param image_shape: The shape that all images will be resized to prior to extraction. If an integer is provided, it applies to both the dimensions.
	:param batch_size: The extractor will process images in batches of this size. Use as big a value as your machine can handle. A power of two is preferred.
	:param num_workers: Number of CPU cores to use for loading images from disk.
	:return: A dictionary with two keys, 'train' and 'val' that returns DataLoaders for the training and validation sets respectively.
	"""
	if image_shape is None and batch_size != 1:
		import warnings
		batch_size = 1
		warnings.warn('Since you wish to use variable image sizes, setting batch_size=1.'
					  '\nDealing with variable inputs is not trivial.', RuntimeWarning)

	return {mode: _get_extract_dataloader(data_path / mode, image_shape, batch_size, num_workers) for mode in ('train', 'val')}

def _get_extract_dataloader(data_path, image_shape=None, batch_size=1, num_workers=0):
	from torch.utils.data.dataloader import DataLoader
	transform = get_transform(image_shape)

	dataset = CocoCaptions(data_path, data_path / 'captions.json', transform)
	return DataLoader(dataset, batch_size, num_workers=num_workers)

def get_transform(image_shape=None):
	"""
	The images go through this pipeline before extraction.

	:param image_shape: The shape that all images will be resized to prior to extraction. If an integer is provided, it applies to both the dimensions.
	:return: TorchVision transform instance which can be applied to any PIL image.
	"""
	from torchvision import transforms

	normalization = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
	transform = [transforms.ToTensor(), transforms.Normalize(**normalization)]
	if image_shape is not None:
		if isinstance(image_shape, int): image_shape = (image_shape, image_shape)
		transform.insert(0, transforms.Resize(image_shape))

	return transforms.Compose(transform)

def get_training_dataloaders(data_path, caption_idx, shuffle):
	"""
	Makes PyTorch DataLoaders for training.

	:param data_path: The root path of the COCO dataset.
	:param caption_idx: The index of the captions to be used for training. The COCO dataset has 5 captions per image. While training, one of these is chosen according to
						the value of this parameter. If negative, indices will be randomly chosen at runtime for each image.
	:param shuffle: Whether to shuffle the dataset while training.
					Shuffling generally produces better performance since the model can't overfit to specific batches.
	:return: A dictionary with two keys, 'train' and 'val' that returns DataLoaders for the training and validation sets respectively.
	"""
	return {mode: _get_training_dataloader(data_path / mode, caption_idx, shuffle) for mode in ('train', 'val')}

def _get_training_dataloader(data_path, caption_idx, shuffle):
	from torch.utils.data.dataloader import DataLoader

	dataset = CocoExtracted(data_path, data_path / 'captions.json', data_path / 'features.pt', caption_idx)
	return DataLoader(dataset, batch_size=1, shuffle=shuffle)

class CocoExtracted(CocoCaptions):
	def __init__(self, root, annFile, feature_file, caption_idx):
		"""
		The COCO Dataset that has the extracted features and the corresponding captions.
		The fundamental difference between this and the CocoCaptions class in torchvision.datasets
		are as follows:

		1. The extracted feature is returned instead of the image.
		2. The captions are preprocessed and only one of them is chosen according to caption_idx.

		:param root: The root path of the COCO dataset.
		:param annFile: The path where the captions JSON file resides.
		:param feature_file: The path where the features were extracted.
		:param caption_idx: The index of the captions to be used for training. The COCO dataset has 5 captions per image. While training, one of these is chosen according to
							the value of this parameter. If negative, indices will be randomly chosen at runtime for each image.
		"""
		import torch

		super().__init__(root, annFile)
		self.features = torch.load(feature_file, map_location='cpu')
		self.caption_idx = caption_idx

	def __getitem__(self, index):
		caption = self._get_caption(index)

		# Add START(`) and END(.) tokens
		caption = '` ' + caption
		if caption[-1] != '.': caption += '.'

		features = self.features[index]
		return features, caption

	def _get_caption(self, index):
		coco = self.coco
		img_id = self.ids[index]
		ann_ids = coco.getAnnIds(imgIds=img_id)
		anns = coco.loadAnns(ann_ids)
		caption_idx = randint(len(anns)) if self.caption_idx < 0 else self.caption_idx

		return anns[caption_idx]['caption']