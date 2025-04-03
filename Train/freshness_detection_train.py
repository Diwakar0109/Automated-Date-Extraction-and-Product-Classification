import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Paths to training and testing directories
train_dir = r"C:\Users\STUDENT\Desktop\expiry_ate\freshness\dataset\train"
test_dir = r"C:\Users\STUDENT\Desktop\expiry_ate\freshness\dataset\test"

# Classes: Including apples, bananas, and oranges
selected_classes = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges']

# Data Augmentation and Normalization2
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=(0.7, 1.3),
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% for validation
)
                                                                                                     
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Training and Validation Generators
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    classes=selected_classes,
    subset='training',
    class_mode='categorical',
    shuffle=True,
    seed=42
)

val_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    classes=selected_classes,
    subset='validation',
    class_mode='categorical',
    shuffle=False,
    seed=42
)

# Testing Generator
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    classes=selected_classes,
    class_mode='categorical',
    shuffle=False
)

# Debugging Information
print(f"Training Samples: {train_gen.samples}")
print(f"Validation Samples: {val_gen.samples}")
print(f"Testing Samples: {test_gen.samples}")
print("Classes Detected:", train_gen.class_indices)

# Load Pretrained ResNet50 Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Model Definition
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),  # First Dropout
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),  # Second Dropout
    Dense(6, activation='softmax', kernel_regularizer=l2(0.001))  # Updated for 6 classes
])

# Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early Stopping Callback
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the Model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[early_stop]
)

# Evaluate Model on Training Data
train_loss, train_acc = model.evaluate(train_gen)
print(f"Training Accuracy: {train_acc * 100:.2f}%, Loss: {train_loss:.4f}")

# Evaluate Model on Testing Data
test_loss, test_acc = model.evaluate(test_gen)
print(f"Test Accuracy: {test_acc * 100:.2f}%, Loss: {test_loss:.4f}")

# Visualize Accuracy and Loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()
plt.show()

# Classification Report for Training Data
train_predictions = np.argmax(model.predict(train_gen), axis=1)
true_train_labels = train_gen.classes
train_class_names = list(train_gen.class_indices.keys())
print("Classification Report (Training):")
print(classification_report(true_train_labels, train_predictions, target_names=train_class_names))

# Classification Report for Testing Data
test_predictions = np.argmax(model.predict(test_gen), axis=1)
true_test_labels = test_gen.classes
test_class_names = list(test_gen.class_indices.keys())
print("Classification Report (Testing):")
print(classification_report(true_test_labels, test_predictions, target_names=test_class_names))

# Confusion Matrix for Testing Data
print("Confusion Matrix (Testing):")
print(confusion_matrix(true_test_labels, test_predictions))

# Save the Model
model.save('model_resnet50_6_classes.h5')
