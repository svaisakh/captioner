import os, shutil

from flask import render_template, request, send_from_directory, redirect, url_for, make_response, jsonify
from app import app
from app.forms import UploadForm
from utils import save_file
from caption import get_captions
from captioner.eval import pretty_print

@app.route('/', methods=['GET', 'POST'])
def index():
	form = UploadForm()
	if not form.validate_on_submit(): return render_template('index.html', form=form)

	filename = save_file(request.files['image'], app.config['UPLOAD_FOLDER'])
	captions = get_captions(url_for('uploaded_file', filename=filename))

	captions = [f'{c} ({p:.2f})' for c, p in captions]
	print(captions)
	return render_template('index.html', captions=captions, image=url_for('uploaded_file', filename=filename), form=form)


@app.route('/caption', methods=['POST'])
def caption():
	filename = save_file(next(request.files.values()), app.config['UPLOAD_FOLDER'])
	captions = get_captions(url_for('uploaded_file', filename=filename))
	return make_response(jsonify({'captions': [{'caption': c, 'p': p} for c, p in captions]}), 200)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
	return send_from_directory(app.config['UPLOAD_FOLDER'],
							   filename)