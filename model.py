import tensorflow as tf
from tensorflow import keras

#Define the model
model = keras.Sequential([
    #Reshape the input to match the dimensions
    keras.layers.Reshape((51, 1), input_shape=(51,)),
    
    #Convolutional layers for feature extraction
    keras.layers.Conv1D(32, 3, activation='relu'),
    keras.layers.MaxPooling1D(2),
    keras.layers.Conv1D(64, 3, activation='relu'),
    keras.layers.MaxPooling1D(2),

    #Recurrent layer to capture temporal patterns
    keras.layers.GRU(64, return_sequences=True),

    #Flatten the output before the dense layers
    keras.layers.Flatten(),

    #Dense layers for classification
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(4, activation='softmax')
])

#Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Print the model summary
model.summary()

model.fit()