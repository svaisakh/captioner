import os, shutil

from werkzeug.utils import secure_filename

def save_file(file, path):
	"""
	Saves the remote uploaded file and returns a secure filename.

	:param file: The uploaded file.
	:param path: Path in which to save.
	:return: The secure filename of the file.
	"""
	filename = secure_filename(file.filename)
	shutil.rmtree(path)
	os.mkdir(path)
	file.save(os.path.join(path, filename))
	return filename