import os
from flask import Flask, render_template, request

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('images')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello_world():
	return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
	file = request.files['image']
	f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

	# add your custom code to check that the uploaded file is a valid image and not a malicious file
	file.save(f)
	
	return render_template('index.html')

