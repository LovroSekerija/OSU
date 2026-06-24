import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt

# CIFAR-10 sadrzi 60,000 slika dimenzija 32x32 piksela u 10 klasa.
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

plt.figure()
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.xticks([]),plt.yticks([])
    plt.imshow(X_train[i])

plt.show()

X_train_n = X_train.astype('float32') / 255.0
X_test_n = X_test.astype('float32') / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = keras.Sequential()
model.add(layers.Input(shape=(32, 32, 3)))

model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2))) # Smanjuje sliku na pola (npr. s 32x32 na 16x16)
model.add(layers.Dropout(0.2))         

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.4))

# Klasifikacijski dio:
model.add(layers.Flatten()) # Pretvara 2D mapu u dugacki 1D niz brojeva
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dropout(0.5)) # Dropout je obicno najjaci na potpuno povezanim slojevima
model.add(layers.Dense(10, activation='softmax')) # Zadnji sloj daje 10 vjerojatnosti

model.summary()

my_callbacks = [
    # Tensorboard (9.4.2): Sprema logove u mapu za vizualizaciju u pregledniku.
    keras.callbacks.TensorBoard(log_dir='logs/cnn_dropout', update_freq=100),
    
    # EarlyStopping (9.4.3): Zaustavlja ucenje ako se preciznost na validacijskom skupu 
    # ne popravi u 5 uzastopnih epoha. To stedi vrijeme i sprjecava overfitting.
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
#optimizer je zapravo learning rate -> adam = 0.001

# Pocinjemo ucenje. Koristimo batch_size=64 sto znaci da mreza vidi 64 slike 
# prije nego sto jednom azurira svoje tezine.
model.fit(X_train_n, y_train, 
          epochs=40, 
          batch_size=64, 
          callbacks=my_callbacks, 
          validation_split=0.1)

# Evaluacija na testnim podacima koje mreza nikada nije vidjela:
score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'\nTocnost na testnom skupu: {100.0*score[1]:.2f}%')
