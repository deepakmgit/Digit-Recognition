import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import Image, ImageDraw

# Load and preprocess the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Model parameters
batch_size = 128
num_classes = 10
epochs = 5
input_shape = (28, 28, 1)

# Build the model
model = Sequential([
    Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, verbose=1),
    ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
]

# Train the model
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=callbacks)

# Evaluate the best model
best_model = tf.keras.models.load_model('best_model.keras')
score = best_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
def predict_digit(img):
    #resize image to 28x28 pixels
    img = img.resize((28,28))
    #convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    img = img.reshape(1,28,28,1)
    img = img/255.01
    img = 1 - img
    #predicting
    res = model.predict(img)[0]
    return np.argmax(res), max(res)

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0

        # Create elements
        self.canvas = tk.Canvas(self, width=800, height=500, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Draw a digit", font=("Helvetica", 24))
        self.classify_btn = tk.Button(self, text="Predict", command=self.classify_handwriting)
        self.clear_btn = tk.Button(self, text="Clear", command=self.clear_all)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2,columnspan=2)
        self.label.grid(row=1, column=0, pady=2, padx=2)
        self.classify_btn.grid(row=2, column=0, pady=2, padx=2)
        self.clear_btn.grid(row=2, column=1, pady=2, padx=2)

        # Draw events
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        self.image1 = Image.new("RGB", (200, 200), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image1)

    def clear_all(self):
        self.canvas.delete("all")
        self.draw.rectangle((0, 0, 200, 200), fill=(255, 255, 255))

    def classify_handwriting(self):
        digit, acc = predict_digit(self.image1)
        self.label.configure(text=str(digit) + ', ' + str(int(acc * 100)) + '%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')
        self.draw.ellipse([self.x - r, self.y - r, self.x + r, self.y + r], fill='black')

app = App()
app.mainloop()
