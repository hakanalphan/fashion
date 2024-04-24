import streamlit as st # to host system on web.
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.svm import SVC



feature_list = np.array(pickle.load(open('featurevector.pkl', 'rb')))
filenames = pickle.load(open('filename.pkl', 'rb'))
image_names = [os.path.basename(file) for file in filenames]

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

# to upload the file - this code is required
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('fashionData', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

# calling the feature extraction function for the uploaded image.
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Content-based filtering recommendation
def content_based_recommendation(features, feature_list, filenames, k=5):
    # Using PCA to reduce dimensionality
    pca = PCA(n_components=100)
    reduced_features = pca.fit_transform(feature_list)
    reduced_query_features = pca.transform(features.reshape(1, -1))
    
    # Calculate cosine similarity
    similarities = cosine_similarity(reduced_query_features, reduced_features)
    
    # Get top k similar images
    indices = similarities.argsort()[0][-k:][::-1]
    return [filenames[i] for i in indices]

# Collaborative filtering recommendation
def collaborative_filtering_recommendation(image_name, similarity_matrix, image_names, k=5):
    index = image_names.index(image_name)
    sim_scores = list(enumerate(similarity_matrix[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:k+1]
    recommended_images = [image_names[i[0]] for i in sim_scores]
    return recommended_images

# kNN recommendation
def knn_recommendation(features, feature_list, filenames, k=5):
    neighbors = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='cosine')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return [filenames[i] for i in indices[0]]
 
def svm_recommendation(features, feature_list, filenames):
    svm = SVC(kernel='linear')
    svm.fit(feature_list, image_names)
    prediction = svm.predict([features])
    index = image_names.index(prediction[0])
    return [filenames[index]]


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display the uploaded file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        
        # Feature extraction
        features = feature_extraction(os.path.join("fashionData", uploaded_file.name), model)
        
        # Content-based recommendation
        st.header("Content-based recommendation")
        recommended_images_content_based = content_based_recommendation(features, feature_list, image_names)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(os.path.join("fashionData", recommended_images_content_based[0]))
        with col2:
            st.image(os.path.join("fashionData", recommended_images_content_based[1]))
        with col3:
            st.image(os.path.join("fashionData", recommended_images_content_based[2]))
        with col4:
            st.image(os.path.join("fashionData", recommended_images_content_based[3]))
        with col5:
            st.image(os.path.join("fashionData", recommended_images_content_based[4]))
        
        # Collaborative filtering recommendation
        st.header("Collaborative filtering recommendation")
        similarity_matrix = cosine_similarity(feature_list)
        recommended_images_collaborative = collaborative_filtering_recommendation(uploaded_file.name, similarity_matrix, image_names)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(os.path.join("fashionData", recommended_images_collaborative[0]))
        with col2:
            st.image(os.path.join("fashionData", recommended_images_collaborative[1]))
        with col3:
            st.image(os.path.join("fashionData", recommended_images_collaborative[2]))
        with col4:
            st.image(os.path.join("fashionData", recommended_images_collaborative[3]))
        with col5:
            st.image(os.path.join("fashionData", recommended_images_collaborative[4]))
        


            

        # kNN recommendation
        st.header("k-Nearest Neighbors (kNN) recommendation")
        recommended_images_knn = knn_recommendation(features, feature_list, filenames)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image( recommended_images_knn[0])
        with col2:
            st.image( recommended_images_knn[1])
        with col3:
            st.image( recommended_images_knn[2])
        with col4:
            st.image( recommended_images_knn[3])
        with col5:
            st.image( recommended_images_knn[4])
    

                       # SVM recommendation
        st.header("Support Vector Machine (SVM) recommendation")
        recommended_images_svm = svm_recommendation(features, feature_list, filenames)
        col1, col2, col3, col4, col5 = st.columns(5)


        with col1:
            st.image(recommended_images_svm[0])
        with col2:
                   st.image(recommended_images_svm[1])
        with col3:
                   st.image(recommended_images_svm[2])
        with col4:
                   st.image(recommended_images_svm[3])
        with col5:
                    st.image(recommended_images_svm[4])
    else:
        st.header("Some error occurred in file upload")