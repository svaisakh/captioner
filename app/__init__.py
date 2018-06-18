from flask import Flask
from config import Config
from caption import get_captions

app = Flask(__name__)
app.config.from_object(Config)

from app import routes