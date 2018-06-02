from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K #TensorFlow


#dimensiones de nuestras imagenes
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 1000
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#creamos el modelo, la red neuronal convolucional
model = Sequential()

#Primera capa de convolución
#filters=32, kernel_size=(3,3) altura y anchura de la ventana de convolución
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
#Función de activación ReLU unidad lineal rectificada
model.add(Activation('relu'))
#Reduce a la mitad los inputs en ambas dimensiones. Forma parte del modelo.
model.add(MaxPooling2D(pool_size=(2, 2)))

#Segunda capa de convolución
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Tercera capa de convolución
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


#Convertimos los mapas de características 3D
#en vectores de características 1D.
#Flatten aplana los imputs. No afecta al tamaño del lote.
model.add(Flatten()) 
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
#Función de activación característica de la curva de aprendizaje
#El aprendizaje se hace de forma más progresiva.
model.add(Activation('sigmoid'))

#Configuramos el proceso de aprendizaje
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#Configuración de "aumentación" que será usada para entrenar
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

#Configuración de "aumentación" que será usada para testear
test_datagen = ImageDataGenerator(rescale=1. / 255)

#Genera los lotes (batches) de datos "aumentados" de las imagenes
#de entreno que van a ser propagados a través de la red
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

#Genera los lotes (batches) de datos "aumentados" de las imagenes
#de testeo/validación que van a ser propagados a través de la red
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

#Entrenamos el modelo sobre los datos generados previamente, lote por lote,
#y para la cantidad de epochs fijada (iteraciones sobre toda la muestra).
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

#Salvamos los pesos del modelo entrenado
model.save_weights('first_try.h5')
