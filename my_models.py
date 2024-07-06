import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from PIL import ImageOps

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

models = {'CNN': [keras.models.load_model('my_model_1.keras'), keras.models.load_model('my_model_1.h5'), pd.read_csv('my_model_1_history.csv')],
          'CNN (based on VGG16)': [keras.models.load_model('my_model_2.keras'), keras.models.load_model('my_model_2.h5'), pd.read_csv('my_model_2_history.csv')]}



def get_models_list():
    return list(models.keys())

def get_model_training_history(name: str):
    return models[name][2]

def get_model(name: str, extention = 'keras'):
    '''extention --> keras or h5'''
    return models[name][0 if extention == 'keras' else 1]

def show_training_history(history):
    '''history: Pandas DF'''
    figure = plt.figure(figsize=(9, 5))
    plt.plot(range(history.shape[0]), history['val_loss'], marker = 'o', markersize = 5, label = 'loss')
    plt.grid(True)
    plt.xlabel('Training Epochs')
    plt.ylabel('loss')

    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.plot(range(history.shape[0]), history['val_acc'], color = 'r', marker = 'o', markersize = 5, label = 'accuracy')
    ax2.set_ylabel('accuracy')
    ax.legend(loc = 'upper left')
    ax2.legend(loc = 'lower left')
    
    return figure

def get_prediction(name: str, image):
    model = get_model(name)

    if name == 'CNN':
        img = image.resize((28, 28))
        img = ImageOps.grayscale(img)
    else:
        img = image.resize((32, 32))
    
    img = np.array(img, np.float32)
    img = img / 255.

    predicted_class = CLASS_NAMES[np.array(tf.argmax(model.predict(np.asarray(img)[None, ...], verbose = False), 1))[0]]
    class_probabilities = pd.DataFrame({'Prob.': [f'{el:.2%}' for el in np.array(tf.nn.softmax(model.predict(np.asarray(img)[None, ...], verbose = False))[0])]},
                                        index = CLASS_NAMES)

    return predicted_class, class_probabilities.T



if __name__ == "__main__":
    print(get_models_list())