import streamlit as st
import my_models as mm
from PIL import Image

# image uploader
uploaded_image = st.sidebar.file_uploader("Select an image...", type=["jpg", "jpeg", "png"])

# models list
models_list = mm.get_models_list()

# select model
model = st.sidebar.selectbox('Select model for classification', models_list)

# show selected model's training history
classification_model_history = mm.get_model_training_history(model)
figure = mm.show_training_history(classification_model_history)
st.sidebar.pyplot(figure)

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    predicted_class, class_probabilities = mm.get_prediction(model, image)

    st.write(f'Predicted Calss: {predicted_class}')
    st.write('Other Calsses Probabilities:')
    st.write(class_probabilities)

    try:
        st.image(image, caption = 'uploaded image', use_column_width = True)
    except Exception as e:
        print(f"Error: {e}")

else:
    st.write('Clothes images classificator')