from flask import Flask, render_template, redirect, url_for, request
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import tensorflow as tf

from models.keras import ModelFactory


import numpy as np
import os
import pandas as pd
from keras.utils import Sequence
from PIL import Image
from skimage.transform import resize

import cv2
from keras import backend as kb

app = Flask(__name__)


class_names=['Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule','Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema','Fibrosis','Pleural_Thickening','Hernia']

model_factory = ModelFactory()
model = model_factory.get_model(class_names,model_name='DenseNet121',use_base_weights=False,weights_path='models/best_weights.h5')
graph = tf.get_default_graph()

def load_image(image_file):
    #image_path = os.path.join(self.source_image_dir, image_file)
    image = Image.open(image_file)
    image_array = np.asarray(image.convert("RGB"))
    image_array = image_array / 255.
    image_array = resize(image_array, (224,224))
    return image_array

def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer

def create_cam(output_dir, image_source, model, class_names):
    #img_ori = cv2.imread(filename=os.path.join(image_source_dir, file_name))
    img_ori = cv2.imread(image_source)
    file_name=image_source[-16:]
    output_path = os.path.join(output_dir, f"{file_name}")

    img_transformed = load_image(image_source)

    # CAM overlay
    # Get the 512 input weights to the softmax.
    class_weights = model.layers[-1].get_weights()[0]
    final_conv_layer = get_output_layer(model, "bn")
    get_output = kb.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([np.array([img_transformed])])
    conv_outputs = conv_outputs[0, :, :, :]
    index=np.argmax(predictions, axis=-1)
    index=index[0][0][0]

    # Create the class activation map.
    cam = np.zeros(dtype=np.float32, shape=(conv_outputs.shape[:2]))
    for i, w in enumerate(class_weights[index]):
        cam += w * conv_outputs[:, :, i]
    # print(f"predictions: {predictions}")
    cam /= np.max(cam)
    cam = cv2.resize(cam, img_ori.shape[:2])
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0
    img = heatmap * 0.5 + img_ori
    cv2.imwrite(output_path, img)

    return output_path

def model_predict(file_path, model):
    #image_path="C://Users/ihrishi/Desktop/files/examples/CheXNet-Keras/data/00000001_001.png"
    image = Image.open(file_path)
    image_array = np.asarray(image.convert("RGB"))
    image_array = image_array / 255.
    image_array = resize(image_array, (224,224))
    #model_factory = ModelFactory()
    test_images=[]
    test_labels=[]
    test_images.append(image_array)
    test_images = np.array(test_images, dtype=np.float32)
    #test_images.shape

    output_dir="static/cam/"

    global graph
    with graph.as_default():
        y=model.predict(test_images,verbose=1)
        op=create_cam(output_dir, file_path, model, class_names)
    #K.clear_session()
    y1=np.argmax(y, axis=-1)
    y1=y1[0][0][0]
    return (op, class_names[y1])

@app.route('/')
def homepage():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads

        basepath = os.path.dirname(__file__)
        #basepath = "C://Users/ihrishi/Desktop/files/examples/CheXNet-Keras/webapp/"
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)


        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #if (preds[0]==1):
        	#result="malignant"
        #else:
        	#result="benign"
        ##return result
        ##result="benign"
        #return result
        #return preds
        return render_template("report.html", cam_image=preds[0][-16:], result=preds[1] )
    return None


if __name__ == "__main__":
    app.debug = True
    app.run()