import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from PIL import Image

st.set_page_config(page_title="Digit Classifier")

st.title("Handwritten Digit Classifier")

@st.cache_resource
def train_model():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test), verbose=0)

    return model

if "ready" not in st.session_state:
    st.session_state.ready = False

if st.button("Train Model"):
    model = train_model()
    st.session_state.model = model
    st.session_state.ready = True
    st.success("Model trained")

if st.session_state.ready:

    uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        image = image.resize((28, 28))
        img_array = np.array(image) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = st.session_state.model.predict(img_array)
        predicted_class = np.argmax(prediction)

        st.image(image, caption="Input Image", width=150)
        st.subheader(f"Predicted Digit: {predicted_class}")

else:
    st.info("Train the model first")