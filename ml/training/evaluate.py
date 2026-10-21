import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

MODEL_PATH = "ml/saved_models/waste_classifier.keras"
TEST_DIR = "ml/data/processed/test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = test_dataset.class_names

print("\nClasses:")
for i, class_name in enumerate(class_names):
    print(i, "->", class_name)

print("\nEvaluating model...")

loss, accuracy = model.evaluate(test_dataset, verbose=1)

print("\n==============================")
print("TEST RESULTS")
print("==============================")
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Accuracy (%): {accuracy * 100:.2f}%")

y_true = []
y_pred = []

for images, labels in test_dataset:
    predictions = model.predict(images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)

print("==============================")
print("CONFUSION MATRIX")
print("==============================")

cm = confusion_matrix(y_true, y_pred)

print(cm)

print("\nEvaluation completed successfully.")
