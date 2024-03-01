import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Veri seti dizinleri
data_dir = 'C:/Users/Acer/OneDrive/Masaüstü/mobilya'
# Oda kategorileri
categories = [ 'oturmaodasi', 'mutfak',"yemekodası","yemekodasıyeni"]

# Veri setini oluşturun
for category in categories:
    for i in range(1, 14):
        img_path = f"{data_dir}/{category}_{i}.jpg"

# Veri setini yüklemek ve ön işlemek için ImageDataGenerator kullanma
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

batch_size = 8
img_size = (224, 224)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(len(categories), activation='softmax')) 


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
logits_size=[8,4]
labels_size=[8,3]
history = model.fit(
    
    train_generator,
    epochs=10,
    validation_data=validation_generator
   
)


plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
