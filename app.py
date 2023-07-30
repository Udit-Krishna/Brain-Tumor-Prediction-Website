from flask import Flask
from flask import request
from flask import render_template
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.utils import load_img,img_to_array
import os
import pickle

model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__, static_url_path='/static')

@app.route('/', methods=['POST','GET'])
def send():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    print(request.files)
    image = request.files["file"]
    image.save('static/predict.jpg')

    test_image = load_img('static/predict.jpg',target_size=(64,64))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis=0)
    result = model.predict(test_image)
    print(result)
    os.remove("static/predict.jpg")
    
    if result[0][0]==0:
        prediction = 'There is tumor'
        print(prediction)
        return render_template('yes.html')
    else :
        prediction = 'There is no tumor'
        print(prediction)
        return render_template('no.html')



if __name__=="__main__":
    app.run(debug=True)