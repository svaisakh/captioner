import os, shutil

from flask import render_template, request, send_from_directory, redirect, url_for, make_response, jsonify
from app import app, get_captions
from app.forms import UploadForm
from utils import save_file

@app.route('/', methods=['GET', 'POST'])
def index():
	form = UploadForm()
	if not form.validate_on_submit(): return render_template('index.html', form=form)

	image = request.files['image']
	filename = save_file(image, app.config['UPLOAD_FOLDER'])
	captions = get_captions(os.path.join(app.config['UPLOAD_FOLDER'], filename))

	captions = [f'{c} ({p:.2f})' for c, p in captions]
	print(url_for('uploaded_file', filename=filename))
	return render_template('index.html', captions=captions, image=url_for('uploaded_file', filename=filename), form=form)


@app.route('/caption', methods=['POST'])
def caption():
	image = next(request.files.values())
	captions = get_captions(image)
	return make_response(jsonify({'captions': [{'caption': c, 'p': p} for c, p in captions]}), 200)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
	return send_from_directory(app.config['UPLOAD_FOLDER'],
							   filename)