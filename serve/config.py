import os

from pathlib import Path

class Config:
	SECRET_KEY = os.environ.get('SECRET_KEY') or 'supposed-to-be-secret'
	UPLOAD_FOLDER = str(Path(__file__).resolve().parent / 'app' / 'uploads')