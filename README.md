#Programa que distingue si una imagen es de un gato o de un perro

El siguiente proyecto de deep learning consiste en la implementación de tres módulos escritos en Python, siguiendo el tutorial:
   
   https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

Generaremos los módulos:

- catvsdog1.py
  
- catvsdog2mv.py

- catvsdog3mv.py

##Requerimientos del proyecto:

- La versión de Python que hemos utilizado es la 3.6
- Necesitamos las siguientes librerias para ejecutar el proyecto: 
    
    - keras
    - tensorflow
    - Pillow
  
  Nota: utilizando PyCharm o Intellij IDEA estas librerias se pueden instalar sin necesidad de utilizar la instrucción:
        
        python -m pip install package
       
  donde package=keras, package=tensorflow, o package=Pillow
  
- Necesitamos crear una carpeta nombrada "data" y las subcarpetas dentro de la primera, nombradas "train" y "validation", respectivamente. 
Además dentro de estas subcarpetas crearemos las 2 subcarpetas "cats" y "dogs". Los archivos son jpgs y nombrados "catXXX" y "dogXXX" respectivamente.
Estas imágenes se han obtenido en 
   
        https://www.kaggle.com/c/dogs-vs-cats/data

En la subcarpeta "cats" de la carpeta "train" hay 1000 imagenes de gatos numeradas de XXX=000 hasta 999. En la subcarpeta dogs
de la carpteta "train" tendremos análogamente 1000 imagenes de perros. En cuanto a las subcarpetas "cats" y "dogs" de la carpeta
"validation" la cantidad de gatos y perros es respectivamente de 500 imagenes númeradas de XXX=000 hasta 499. Observad que las imagenes
serán nombradas igual que en la carpeta "train". 
- Para poder ejecutar el módulo "catvsdog3mv.py" necesitamos el archivo de pesos del modelo pre-entrenado VGG16:
      
      vgg16_weight.h5  
 
  y lo guardaremos en una carpeta que llamaremos "examples" que estará a su vez en la carpeta "keras" que a su vez
  estará alojada en la carpeta de todos los proyectos que tenemos creados.

##Modo de ejecutar el proyecto y resultados que esperamos encontrar:

Para ejecutar el proyecto lo único que hemos de ejecutar es cada uno de los tres módulos.
Por supuesto que hemos de verificar que los módulos accederán a:
   
   - CatvsDog/data/train/cats
   - CatvsDog/data/train/dogs
   - CatvsDog/data/validation/cats
   - CatvsDog/data/validation/dogs
 
 donde "CatvsDog" es el nombre que le hemos dado a la carpeta de nuestro proyecto, donde además tenemos también los tres 
 módulos que hemos implementado con el lenguaje Python.
 
 Por otro lado, también hemos de verificar que tenemos la ruta:
 
   - keras/examples
 
 donde como hemos dicho anteriormente están los pesos del modelo pre-entrenado VGG16.
 
 Los resultados esperados son los siguientes:
 
 - Con el módulo "catvsdog1.py", el final de su ejecución será el archivo de pesos
 
       first_try.h5
 
 - Con el módulo "catvsdog2mv.py", el final de su ejecución será el archivo de pesos
      
       bottleneck_fc_model.h5
 
 - Con el módulo "catvsdog3mv.py", el final de su ejecución será el archivo de pesos
     
       fine_tune_fc_model.h5
       
 Lo importante de la ejecución de estos tres módulos es que mejoramos la capacidad de 
 clasificación de la red neuronal con cada uno. Es decir, con el primero creamos una 
 red neuronal con pocas capas de convolución, lo que hace que sea el peor de los tres. Con 
 el segundo introducimos el modelo VGG16 con 16 capas más una parte adicional final formada por una 
 modelo completamente conectado. De esta manera la red neuronal puede captar características
 ocultas que no era capaz de detectar con el primer módulo. Así mejoramos su poder de clasificación. 
 No obstante, como entrenamos toda la red neuronal a través de todas las capas del modelo, requiere 
 en principio un mayor tiempo de ejecución. Finalmente, en el tercer módulo, aplicamos el hecho de 
 aprovechar que el modelo VGG16 ya ha sido pre-entrenado en otros problemas de clasificación y ha 
 podido captar características útiles para diferenciar imagenes. Esto podría no funcionar bien. Aunque en
 la práctica ha dado buenos resultados. En particular, en el problema de clasifición de imagenes de gatos y
 perros, es lo que se espera que pase. Además también cargamos en el modelo los pesos que hemos obtenido con
 el segundo módulo de Python. Así, después de cargar el módelo VGG16 y los pesos del archivo
 "vgg16_weights.h5", lo que hacemos es sólamente entrenar la parte adicional del modelo con
 imagenes de perros y gatos. En cierto sentido, con el tercer módulo afinamos los resultados obtenidos con el segundo.
        

