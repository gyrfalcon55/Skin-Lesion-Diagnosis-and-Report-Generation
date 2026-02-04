import cv2
import numpy as np
import tensorflow as tf

# Input size expected by the U-Net model
IMG_SIZE = (128, 128)

# Load the segmentation model once
segmentation_model = tf.keras.models.load_model(
    "models/unet_model(128).h5"
)


def segment_lesion(image):
    """
    Segment the lesion region from the input image using a U-Net model.

    Parameters:
    image : Input image in BGR format (OpenCV)

    Returns:
    segmented_image : Original image masked by the predicted lesion region
    mask            : Binary segmentation mask resized to original image size
    """

    # Resize image to U-Net input size and normalize
    resized_image = cv2.resize(image, IMG_SIZE)
    normalized_image = resized_image / 255.0

    # Add batch dimension
    input_tensor = np.expand_dims(normalized_image, axis=0)

    # Predict segmentation mask
    predicted_mask = segmentation_model.predict(input_tensor, verbose=0)[0]

    # Convert probability map to binary mask
    binary_mask = (predicted_mask > 0.5).astype(np.uint8)

    # Resize mask back to original image size
    binary_mask = cv2.resize(
        binary_mask,
        (image.shape[1], image.shape[0])
    )

    # Apply mask to the original image
    segmented_image = image * binary_mask[:, :, None]

    return segmented_image, binary_mask
