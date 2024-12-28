import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from tensorflow.keras.utils import to_categorical

#create dataset and set labels.
def load_dataset(file_path):
    data = pd.read_csv(file_path)
    inputs = data.values
    label = file_path.split('_')[1].split('.')[0]
    encoder = LabelEncoder()
    encoder.fit(['empty', 'walking', 'sitting', 'standing'])
    label = encoder.transform([label])[0]   
    label = to_categorical(label, num_classes=4)
    return inputs, np.tile(label, (len(inputs), 1))

#specify input files.
input_files = ['input_walking.csv', 'input_sitting.csv', 'input_standing.csv', 'input_empty.csv']
inputs = []
labels = []
for file_path in input_files:
    data, label = load_dataset(file_path)
    inputs.append(data)
    labels.append(label)
inputs = np.concatenate(inputs, axis=0)
labels = np.concatenate(labels, axis=0)

#PCA transoformation, select component number.
pca = PCA(n_components=10) 
inputs_pca = pca.fit_transform(inputs)

#normalise the data.
scaler = MinMaxScaler()
inputs_scaled = scaler.fit_transform(inputs_pca)

#CNN model.
model = keras.Sequential([
    keras.layers.Reshape((10, 1), input_shape=(10,)),  # Adjust the input shape to the number of components
    keras.layers.Conv1D(8, 3, activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(4, activation='softmax')
])

#compile model.
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#training:testing:validation ratio = 55:20:25.
X_train, X_test, y_train, y_test = train_test_split(inputs_scaled, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

#model application/evaluation.
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")
model.summary()

#plot accuracy.
epochs = range(1, 51)
test_accuracy = history.history['val_accuracy']
plt.plot(epochs, test_accuracy, 'b-o')
plt.xlabel('Epochs')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy vs. Epochs')
plt.grid(True)
plt.show()
