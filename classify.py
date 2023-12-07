import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

# Set the root directory containing subfolders for each celebrity
root_dir = 'cropped'

celebrities = os.listdir(root_dir)
num_classes = len(celebrities)


dataset = []
label = []
img_size = (128, 128)

for i, celebrity_name in tqdm(enumerate(celebrities), desc="Loading Data"):
    celebrity_path = os.path.join(root_dir, celebrity_name)
    celebrity_images = os.listdir(celebrity_path)
    
    for image_name in celebrity_images:
        if image_name.split('.')[1] == 'png':
            image = cv2.imread(os.path.join(celebrity_path, image_name))
            image = Image.fromarray(image, 'RGB')
            image = image.resize(img_size)
            dataset.append(np.array(image))
            label.append(i)

dataset = np.array(dataset)
label = np.array(label)

print("--------------------------------------\n")
print('Dataset Length: ', len(dataset))
print('Label Length: ', len(label))
print("--------------------------------------\n")

print("--------------------------------------\n")
print("Train-Test Split")
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=42)
print("--------------------------------------\n")

print("--------------------------------------\n")
print("Normalizing the Dataset. \n")

# Normalize the images
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

print("--------------------------------------\n")

# ... (the rest of your code remains mostly the same)

# Update the output layer in your model
output_classes = num_classes  # Change this according to the number of celebrities
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(output_classes, activation='softmax')] ) 
print("--------------------------------------\n")
model.summary()
print("--------------------------------------\n")

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=200, batch_size=128, validation_split=0.1)

plt.plot(history.epoch,history.history['accuracy'], label='accuracy')
plt.plot(history.epoch,history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig(r'C:\Users\ACER\Desktop\dataset_celebrities\celebrity_accuracy_plot.png')
plt.savefig(r'')

plt.clf()


plt.plot(history.epoch,history.history['loss'], label='loss')
plt.plot(history.epoch,history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig(r'C:\Users\ACER\Desktop\dataset_celebrities\celebrity_sample_loss_plot.png')


print("--------------------------------------\n")
print("Model Evaluation Phase.\n")
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Accuracy: {round(accuracy * 100, 2)}')
print("--------------------------------------\n")

# Model Prediction
def make_prediction(img, model, celebrities):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = Image.fromarray(img)
    img = img.resize((128, 128))
    img = np.array(img)
    input_img = np.expand_dims(img, axis=0)
    input_img = tf.keras.utils.normalize(input_img, axis=1)  # Normalize the image
    predictions = model.predict(input_img)
    predicted_class = np.argmax(predictions)
    celebrity_name = celebrities[predicted_class]
    print(f"Predicted Celebrity: {celebrity_name}")

# ... (the rest of your code remains mostly the same)

# Make predictions
make_prediction(os.path.join(root_dir, 'virat_kohli', 'virat_kohli4.png'), model, celebrities)
make_prediction(os.path.join(root_dir, 'roger_federer', 'roger_federer8.png'), model, celebrities)
make_prediction(os.path.join(root_dir, 'maria_sharapova', 'maria_sharapova11.png'), model, celebrities)
make_prediction(os.path.join(root_dir, 'lionel_messi', 'lionel_messi15.png'), model, celebrities)
make_prediction(os.path.join(root_dir, 'serena_williams', 'serena_williams22.png'), model, celebrities)
