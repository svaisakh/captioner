from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from wtforms import SubmitField

class UploadForm(FlaskForm):
	image = FileField('image', validators=[FileRequired()])
	submit = SubmitField('Submit')