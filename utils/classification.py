import tensorflow as tf
import numpy as np
from utils.preprocessing import preprocess_image


# Load the classification model once
classification_model = tf.keras.models.load_model(
    "models/efficientnetB1.h5"
)

# Class labels corresponding to model outputs
CLASS_NAMES = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis-like lesions",
    "Dermatofibroma",
    "Melanocytic nevi",
    "Melanoma",
    "Squamous cell carcinoma",
    "Vascular lesions"
]


def classify(image):
    """
    Classify a skin lesion image using the EfficientNet model.

    Parameters:
    image : Input image in BGR format (OpenCV)

    Returns:
    predicted_class : Name of the predicted lesion type
    confidence      : Prediction confidence score
    """

    # Preprocess image for EfficientNet
    input_tensor = preprocess_image(image)

    # Run model inference
    predictions = classification_model.predict(
        input_tensor, verbose=0
    )[0]

    # Get predicted class index
    predicted_index = np.argmax(predictions)

    return (
        CLASS_NAMES[predicted_index],
        float(predictions[predicted_index])
    )
