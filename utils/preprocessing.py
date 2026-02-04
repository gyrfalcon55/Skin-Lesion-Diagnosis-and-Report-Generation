import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

# Expected input size for EfficientNet
IMG_SIZE = (224, 224)


def preprocess_image(image):
    """
    Preprocess an image for EfficientNet classification.

    Parameters:
    image : Input image in BGR format (OpenCV)

    Returns:
    A NumPy array of shape (1, 224, 224, 3) ready for model inference
    """

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    image = cv2.resize(image, IMG_SIZE)

    # Apply EfficientNet-specific preprocessing
    image = preprocess_input(image)

    # Add batch dimension
    return np.expand_dims(image, axis=0)
