import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 1000
epochs = 50
batch_size = 16


#Construimos la red neuronal del modelo VGG16
model = applications.VGG16(include_top=False, weights='imagenet')
print('Model loaded.')

#Configuración de "aumentación" que será usada en la
#generación de imagenes
datagen = ImageDataGenerator(rescale=1. / 255)

#Genera los lotes (batches) de datos "aumentados" de las imagenes
#de entreno que van a ser propagados a través de la red
#antes de las capas completamente conectadas.
#class_mode=None, porque el generador sólo producirá lotes de datos, sin etiquetas
#shuffle=False, porque los datos estarán en orden, por lo que las primeras
#1000 imágenes serán gatos y después 1000 perros.
generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)
print('generator train.')
#el método predict_generator devuelve predicciones del modelo anterior, a partir
#de los lotes de entrenamiento que hemos generado produciendo lotes de datos numpy.
bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
#la salida se guarda como un array Numpy
np.save(open('bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)
print('bottleneck_features_train.')

#Genera los lotes (batches) de datos "aumentados" de las imagenes
#de testeo/validación que van a ser propagados a través de la red
#antes de las capas completamente conectadas.
#class_mode=None, porque el generador sólo producirá lotes de datos, sin etiquetas
#shuffle=False, porque los datos estarán en orden, por lo que las primeras
#500 imágenes serán gatos y después 500 perros.
generator = datagen.flow_from_directory(
   validation_data_dir,
   target_size=(img_width, img_height),
   batch_size=batch_size,
   class_mode=None,
   shuffle=False)
print('generator validation.')
#el método predict_generator devuelve predicciones del modelo anterior, a partir
#de los lotes de validacion que hemos generado produciendo lotes de datos numpy.
bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
#la salida se guarda como un array Numpy
np.save(open('bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)
print('bottleneck_features_validation.')


#Ahora podemos entrenar la parte del modelo que es completamente conectada.
#Cargamos los valores de las caracteristicas obtenidas en el entreno previo.
train_data = np.load(open('bottleneck_features_train.npy'))
print('train_data.')
train_labels = np.array(
        [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))
print('train_labels.')

#Cargamos los valores de las caracteristicas obtenidas en la validación previa.
validation_data = np.load(open('bottleneck_features_validation.npy'))
print('validation_data.')
validation_labels = np.array(
        [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))
print('train_labels.')

#Convertimos los mapas de características 3D
#en vectores de características 1D.
#Flatten aplana los imputs. No afecta al tamaño del lote.
#Necesitamos especificar el input_shape porque no lo hemos hecho antes.
model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
print('establecemos el modelo que entrenaremos sobre las características almacenadas.')

#Configuramos el proceso de aprendizaje
model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])
print('configuración del proceso de aprendizaje.')
    
#Entrenamos el modelo sobre los datos generados previamente, lote por lote,
#y para la cantidad de epochs fijada (iteraciones sobre toda la muestra).
model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
print('Aprendizaje finalizado.')

#Salvamos los pesos del modelo entrenado
model.save_weights('bottleneck_fc_model.h5')
print('Fin del programa.')

