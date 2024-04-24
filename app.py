import tensorflow
# image module to access functionalities related to images.
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


# print(model.summary())

def extract_features(img_path, model):
    # loading the image
    img = image.load_img(img_path, target_size=(224, 224))

    # converting the image to an array - 224 * 224 * 3 - since RGB image
    img_array = image.img_to_array(img)

    # to reshape the images - input= image array , output=batch of images.
    expanded_img_array = np.expand_dims(img_array, axis=0)

    
    preprocessed_img = preprocess_input(expanded_img_array)

    result = model.predict(preprocessed_img).flatten()

    # normalize the value from 0 to 1
    normalized_result = result / norm(result)

    return normalized_result


filenames = []

for file in os.listdir('fashionData'):
    filenames.append(os.path.join('fashionData', file))

feature_list = []

for file in tqdm(filenames):
    feature_list.append(extract_features(file, model))

pickle.dump(feature_list, open('featurevector.pkl', 'wb'))
pickle.dump(filenames, open('filename.pkl', 'wb'))