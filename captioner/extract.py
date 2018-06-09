import magnet as mag

def Extractor(architecture):
	import torchvision.models

	architecture = getattr(torchvision.models, architecture)
	model = architecture(pretrained=True).to(mag.device)
	_detach_head(model)

	return model

def _detach_head(model):
	from types import MethodType
	from torch.nn import AdaptiveAvgPool2d
	from torch.utils.data.dataloader import DataLoader

	def extractor_forward(self, x):
		if isinstance(x, DataLoader):
			from torch import cat

			from captioner.utils import get_tqdm
			tqdm = get_tqdm()

			return cat([self(x_i.to(mag.device)) for x_i, _ in tqdm(iter(x))])

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)

		return x

	model.feature_size = model.fc.in_features
	del model.fc
	model.avgpool = AdaptiveAvgPool2d(1)
	model.forward = MethodType(extractor_forward, model)

def __main(architecture, image_shape, extractor_batch_size, num_workers):
	import torch

	from pathlib import Path
	from captioner.data import get_extract_dataloaders

	DIR_DATA = Path('~/.data/COCO').expanduser()

	dataloader = get_extract_dataloaders(DIR_DATA, image_shape, extractor_batch_size, num_workers)
	extractor = Extractor(architecture)

	for mode, name in (('val', 'Validation'), ('train', 'Training')):
		print(f'Extracting features for {name} set.')
		with mag.eval(extractor): features = extractor(dataloader[mode])
		torch.save(features, DIR_DATA / mode / 'features.pt')

	print('Done')

if __name__ == '__main__':
	import hparams

	from captioner.utils import launch

	launch(__main, default_module=hparams)