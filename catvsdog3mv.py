from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense

#Ruta donde están los pesos del modelo pre-entrenado
weights_path = 'keras/examples/vgg16_weights.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'
#Dimensiones de nuestras imagenes
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
model.summary()

#Construimos el modelo clasificador para ponerlo en la parte superior
#del modelo de convolución
##Convertimos los mapas de características 3D
#en vectores de características 1D.
#Flatten aplana los imputs. No afecta al tamaño del lote.
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

#Para que el hacer "fine-tuning" (afinar) tenga exito es 
#necesario empezar con un clasificador completamente entrenado,
#incluyendo el "top classifier"
#Así evitamos que las grandes actualizaciones de los grandientes
#desencadenadas por los pesos inicializados aleatóriamente, puedan
#arruinar los pesos aprendidos en la base de convolución.
top_model.load_weights(top_model_weights_path)

#Añadimos el modelo en la parte de arriba de la base de convolución
model.add(top_model)

#Ponemos las primeras 25 capas (las 7 capas del "top classifier" y
#las 18 capas del modelo VGG16 que hay hasta el último bloque de convolución)
#en modo no entrenable (así los pesos no serán adaptados)
#Evitamos así el sobreajuste debido a la alta capacidad entrópica de la red,
#conservando características más generales.
for layer in model.layers[:25]:
    layer.trainable = False

#Configuramos el proceso de aprendizaje.
#Se utiliza un algoritmo de optimización de descendente
#gradiente estocástico y una velocidad de aprendizaje muy lenta.
#Utilizamos un momentum alto para acelerar el optimizador en
#la dirección correspondiente y amortiguar las oscilaciones.
#Se consigue que la magnitud de las actualizaciones de los pesos
#sea muy pequeña, para no estropear las características aprendidas
#previamente.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
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
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

#Genera los lotes (batches) de datos "aumentados" de las imagenes
#de testeo/validación que van a ser propagados a través de la red
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

#Entrenamos el modelo sobre los datos generados previamente, lote por lote,
#y para la cantidad de epochs fijada (iteraciones sobre toda la muestra).
#Aquí es donde se hace el "fine-tune" del modelo.
model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)

#Salvamos los pesos del modelo entrenado
model.save_weights('fine_tune_fc_model.h5')
