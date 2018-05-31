import magnet as mag

def Extractor(architecture):
	import torchvision.models

	architecture = getattr(torchvision.models, architecture)
	model = architecture().to(mag.device)
	_detach_head(model)

	return model

@mag.eval
def extract(extractor, dataloader):
	import torch

	from utils import get_tqdm
	tqdm = get_tqdm()
	
	batch_size = dataloader.batch_size
	num_images = len(dataloader.dataset)
	feature_size = _feature_size(extractor, dataloader)

	features = torch.zeros(num_images, feature_size).to(mag.device)

	for i, (x, _) in enumerate(tqdm(iter(dataloader))):
		y = extractor(x.to(mag.device))
		features[i * batch_size: min((i + 1) * batch_size, num_images)] = y

	return features

def get_dataloaders(data_path, image_shape, batch_size):
	from torchvision import transforms

	normalization = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
	transform = transforms.Compose([transforms.Resize(image_shape), transforms.ToTensor(), transforms.Normalize(**normalization)])

	return {mode: _get_dataloader(data_path / mode, transform, image_shape, batch_size) for mode in ('train', 'val')}

def _get_dataloader(data_path, transform, image_shape, batch_size):
	from torch.utils.data.dataloader import DataLoader
	from torchvision.datasets import CocoCaptions

	dataset = CocoCaptions(data_path, data_path / 'captions.json', transform)
	return DataLoader(dataset, batch_size)

@mag.eval
def _feature_size(extractor, dataloader):
    return len(extractor(next(iter(dataloader))[0][:1].to(mag.device))[0])

def _detach_head(model):
	from types import MethodType

	def extractor_forward(self, x):
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

	del model.fc  
	model.forward = MethodType(extractor_forward, model)

def __main():
	from pathlib import Path
	from hparams import image_shape, architecture
	from hparams import extractor_batch_size as batch_size

	DIR_DATA = Path('~/.data/COCO').expanduser()

	dataloader = get_dataloaders(DIR_DATA, image_shape, batch_size)
	extractor = Extractor(architecture)

	for mode in ('val', 'train'):
		print(f'Extracting features for set {mode}')
		features = extract(extractor, dataloader[mode])
		torch.save(features.to('cpu'), DIR_DATA / mode / 'features.pt')
		
if __name__ == '__main__': __main()