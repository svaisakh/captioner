from pathlib import Path
from captioner.utils import working_directory

image_url = lambda mode: f'http://images.cocodataset.org/zips/{mode}2017.zip'
annotations_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'

def download_images(mode, path):
    download_and_extract(image_url(mode), path, extract_path=mode)

def download_captions(path):
    if Path(path / 'train/captions.json').exists(): return # Why bother. Job already done

    def extras():
        import shutil

        for mode in ('train', 'val'):
            os.rename(f'annotations/captions_{mode}2017.json', f'{mode}/captions.json')
        shutil.rmtree('annotations')

    download_and_extract(annotations_url, path, extras=extras)

@working_directory
def download_and_extract(url, path, extract_path=None, extras=None):
    import wget, os

    from zipfile import ZipFile

    url = Path(url)
    filename = url.name

    if extract_path is not None and Path(extract_path).exists():
        return # Why bother. Job already done

    # Download if not yet done
    if not Path(filename).exists():
        print('Downloading...')
        wget.download(url)

    print('Extracting...')
    ZipFile(filename).extractall(extract_path) # Extract

    os.remove(filename) # Remove the zip file since it's no longer needed

    if extras is not None: extras()

def __main():
	DIR_DATA = Path('~/.data/COCO').expanduser()

	for name, mode in zip(('Training', 'Validation'), ('train', 'val')):
		print(f'{name} Images')
		download_images(mode, DIR_DATA)

	print('Captions')
	download_captions(DIR_DATA)

if __name__ == '__main__': __main()

