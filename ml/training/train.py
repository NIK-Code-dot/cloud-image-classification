import os
import tensorflow as tf
tf.config.optimizer.set_jit(False)
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# =========================
# Configuration
# =========================

TRAIN_DIR = "ml/data/processed/train"
VAL_DIR = "ml/data/processed/val"
MODEL_DIR = "ml/saved_models"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10

os.makedirs(MODEL_DIR, exist_ok=True)


# =========================
# Load datasets
# =========================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=42
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)


class_names = train_dataset.class_names

print("\nClasses:")
for index, class_name in enumerate(class_names):
    print(index, "->", class_name)

print("\nNumber of classes:", len(class_names))


# =========================
# Performance optimization
# =========================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
val_dataset = val_dataset.prefetch(AUTOTUNE)


# =========================
# Data augmentation
# =========================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])


# =========================
# Base model
# =========================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False


# =========================
# Build model
# =========================

inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.2)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)


# =========================
# Compile
# =========================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# =========================
# Callbacks
# =========================

checkpoint_path = os.path.join(
    MODEL_DIR,
    "waste_classifier.keras"
)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True
    )
]


# =========================
# Training
# =========================

print("\nStarting model training...\n")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)


# =========================
# Save final model
# =========================

final_model_path = os.path.join(
    MODEL_DIR,
    "waste_classifier_final.keras"
)

model.save(final_model_path)

print("\nTraining completed successfully.")

print("Best model saved at:")
print(checkpoint_path)

print("\nFinal model saved at:")
print(final_model_path)
