import tensorflow as tf
import numpy as np

MODEL_PATH = "ml/saved_models/waste_classifier.keras"

model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation"
]

IMAGE_SIZE = (224, 224)


def predict_image(image):
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image)

    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]

    image_array = np.expand_dims(image_array, axis=0)

    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    predictions = model.predict(image_array, verbose=0)

    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_index])

    predicted_class = CLASS_NAMES[predicted_index]

    return predicted_class, confidence
