from torchvision.datasets import CocoCaptions

def get_extract_dataloaders(data_path, image_shape, batch_size, num_workers):
    transform = get_transform()

    return {mode: _get_extract_dataloader(data_path / mode, transform, image_shape, batch_size, num_workers) for mode in ('train', 'val')}

def _get_extract_dataloader(data_path, transform, image_shape, batch_size, num_workers):
    from torch.utils.data.dataloader import DataLoader

    dataset = CocoCaptions(data_path, data_path / 'captions.json', transform)
    return DataLoader(dataset, batch_size, num_workers=num_workers)

def get_transform(image_shape):
    from torchvision import transforms

    normalization = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    return transforms.Compose([transforms.Resize(image_shape), transforms.ToTensor(), transforms.Normalize(**normalization)])

def get_training_dataloaders(data_path, caption_idx, shuffle):
    return {mode: _get_training_dataloader(data_path / mode, caption_idx, shuffle) for mode in ('train', 'val')}

def _get_training_dataloader(data_path, caption_idx, shuffle):
    from torch.utils.data.dataloader import DataLoader

    dataset = CocoExtracted(data_path, data_path / 'captions.json', data_path / 'features.pt', caption_idx)

    return DataLoader(dataset, batch_size=1, shuffle=shuffle)

class CocoExtracted(CocoCaptions):
    def __init__(self, root, annFile, feature_file, caption_idx):
        import torch

        super().__init__(root, annFile)
        self.features = torch.load(feature_file)
        self.caption_idx = caption_idx
        
    def __getitem__(self, index):
        caption = self._get_caption(index)
        
        # Add START(`) and END(.) tokens
        caption = '` ' + caption
        if caption[-1] != '.': caption += '.'
        
        features = self.features[index]
        return features, caption
    
    def _get_caption(self, index):
        from numpy.random import randint
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        caption_idx = randint(len(anns)) if self.caption_idx is None else self.caption_idx
        
        return anns[caption_idx]['caption']