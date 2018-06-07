import os, shutil

from werkzeug.utils import secure_filename

def save_file(file, path):
	filename = secure_filename(file.filename)
	shutil.rmtree(path)
	os.mkdir(path)
	file.save(os.path.join(path, filename))
	return filename