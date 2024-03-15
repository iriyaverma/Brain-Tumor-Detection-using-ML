import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive')

from sklearn.preprocessing import LabelEncoder

folder_path = '/content/drive/MyDrive/Brain Tumor prj'

train_folder = '/content/drive/MyDrive/Brain Tumor prj/Training'
test_folder = '/content/drive/MyDrive/Brain Tumor prj/Testing'

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth for each GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

import os
files = os.listdir(folder_path)
print(files)

def preprocess_image(file_path):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (224, 224))  # Resize to fit VGG16 input size
    img = img.astype(np.float32) / 255.0  # Normalize pixel values
    return img

def load_images_from_folder(folder):
    images = []
    labels = []
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            label = subfolder
            for filename in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, filename)
                if os.path.isfile(file_path):
                    image = preprocess_image(file_path)
                    images.append(image)
                    labels.append(label)
    return np.array(images), np.array(labels)

X_train, y_train = load_images_from_folder(train_folder)
X_test, y_test = load_images_from_folder(test_folder)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i])
    ax.axis('off')
    ax.set_title(y_train[i])
plt.show()

import tensorflow as tf
from tensorflow.keras import layers, models

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(4, activation='softmax')  # Assuming 4 classes: glioma, meningioma, no tumor, pituitary
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train_encoded, epochs=10, validation_data=(X_val, y_val_encoded))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

loss, accuracy = model.evaluate(X_test, y_test_encoded)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

conf_matrix = confusion_matrix(y_test, y_pred_classes)

precision = precision_score(y_test, y_pred_classes, average='weighted')
recall = recall_score(y_test, y_pred_classes, average='weighted')
f1 = f1_score(y_test, y_pred_classes, average='weighted')

print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print("Confusion Matrix:")
print(conf_matrix)

import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

"""VGG16"""

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)  # Assuming 4 classes

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

history_vgg16 = model.fit(X_train, y_train_encoded, epochs=10, validation_data=(X_val, y_val_encoded))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history_vgg16.history['accuracy'])
plt.plot(history_vgg16.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

plt.subplot(1, 2, 2)
plt.plot(history_vgg16.history['loss'])
plt.plot(history_vgg16.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

loss_vgg16, accuracy_vgg16 = model.evaluate(X_test, y_test_encoded)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Calculate confusion matrix
conf_matrix_vgg16 = confusion_matrix(y_test, y_pred_classes_vgg16)

# Calculate precision, recall, and F1-score
precision_vgg16 = precision_score(y_test, y_pred_classes_vgg16, average='weighted')
recall_vgg16 = recall_score(y_test, y_pred_classes_vgg16, average='weighted')
f1_vgg16 = f1_score(y_test, y_pred_classes_vgg16, average='weighted')

print(f"VGG16 Test Loss: {loss_vgg16}")
print(f"VGG16 Test Accuracy: {accuracy_vgg16}")
print(f"VGG16 Precision: {precision_vgg16}")
print(f"VGG16 Recall: {recall_vgg16}")
print(f"VGG16 F1-Score: {f1_vgg16}")
print("VGG16 Confusion Matrix:")
print(conf_matrix_vgg16)

y_pred_vgg16 = model.predict(X_test)
y_pred_classes_vgg16 = np.argmax(y_pred_vgg16, axis=1)

# Calculate confusion matrix
conf_matrix_vgg16 = confusion_matrix(y_test, y_pred_classes_vgg16)

precision_vgg16 = precision_score(y_test, y_pred_classes_vgg16, average='weighted')
recall_vgg16 = recall_score(y_test, y_pred_classes_vgg16, average='weighted')
f1_vgg16 = f1_score(y_test, y_pred_classes_vgg16, average='weighted')

print(f"VGG16 Test Loss: {loss_vgg16}")
print(f"VGG16 Test Accuracy: {accuracy_vgg16}")
print(f"VGG16 Precision: {precision_vgg16}")
print(f"VGG16 Recall: {recall_vgg16}")
print(f"VGG16 F1-Score: {f1_vgg16}")
print("VGG16 Confusion Matrix:")
print(conf_matrix_vgg16)



final_val_acc_cnn = history.history['val_accuracy'][-1]
final_val_acc_vgg16 = history_vgg16.history['val_accuracy'][-1]

print("Final Validation Accuracy - CNN Model:", final_val_acc_cnn)
print("Final Validation Accuracy - VGG16 Model:", final_val_acc_vgg16)

if final_val_acc_cnn > final_val_acc_vgg16:
    print("CNN Model has a higher validation accuracy.")
else:
    print("VGG16 Model has a higher validation accuracy.")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['val_accuracy'], label='CNN Validation Accuracy')
plt.plot(history_vgg16.history['val_accuracy'], label='VGG16 Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy Comparison')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['val_loss'], label='CNN Validation Loss')
plt.plot(history_vgg16.history['val_loss'], label='VGG16 Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss Comparison')
plt.legend()

plt.show()

final_val_acc_cnn_percent = final_val_acc_cnn * 100
final_val_acc_vgg16_percent = final_val_acc_vgg16 * 100

print("Final Validation Accuracy - CNN Model: {:.2f}%".format(final_val_acc_cnn_percent))
print("Final Validation Accuracy - VGG16 Model: {:.2f}%".format(final_val_acc_vgg16_percent))


