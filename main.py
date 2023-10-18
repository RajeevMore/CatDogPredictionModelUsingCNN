import os
#from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

# libraries for model 

import numpy 
import tensorflow
from tensorflow.keras.models import load_model
import cv2

from flask import Flask
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # use CPU only
UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
# Model saved with Keras model.save()
MODEL_PATH ='myvgg1.h5'

# Load your trained model
model = load_model(MODEL_PATH)

#model predict 
def model_predict(file_path, model):
    A = cv2.imread(file_path)
    A = cv2.resize(A,(224,224))
    A = A/255
    A = A.reshape(1,224,224,3)
    if model.predict(A).argmax()==0: 
        animal = "CAT"
    else:
        animal = "DOG"
    return('Predicted Animal is : '+animal)
    
  
@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    
    # Save file in upload_folder
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
    
        # Make prediction
        result = model_predict(file_path, model)
        print(result)
        flash(result)
        return render_template('upload.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename)) #status code

if __name__ == "__main__":
    app.run(debug=True)
