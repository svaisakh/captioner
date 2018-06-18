import os

from pathlib import Path

class Config:
	SECRET_KEY = os.environ.get('SECRET_KEY') or b'{n\xae7Uu\xb6~Z\x899\xa6\xf1\xc3Y3\x87\x81\x17\x1c9Jx\x97'
	UPLOAD_FOLDER = '/tmp/uploads'#str(Path(__file__).resolve().parent / 'app' / 'uploads')
	Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
