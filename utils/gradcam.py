import tensorflow as tf
import numpy as np
import cv2


def generate_gradcam(model, image, class_index):
    """
    Generate Grad-CAM heatmap for an EfficientNet-based classifier.

    Parameters:
    model       : Full classification model
    image       : Preprocessed image (1, H, W, 3)
    class_index : Index of predicted class
    """

    # --------------------------------------------------
    # Find EfficientNet backbone dynamically
    # --------------------------------------------------
    backbone = None
    for layer in model.layers:
        if "efficientnet" in layer.name.lower():
            backbone = layer
            break

    if backbone is None:
        raise ValueError("EfficientNet backbone not found in model.")

    # --------------------------------------------------
    # Get last convolutional layer inside EfficientNet
    # --------------------------------------------------
    last_conv_layer = backbone.get_layer("top_activation")

    # Model to extract feature maps
    feature_extractor = tf.keras.Model(
        inputs=backbone.input,
        outputs=last_conv_layer.output
    )

    # Layers after backbone (classifier head)
    classifier_layers = model.layers[
        model.layers.index(backbone) + 1 :
    ]

    with tf.GradientTape() as tape:
        conv_output = feature_extractor(image)
        tape.watch(conv_output)

        x = conv_output
        for layer in classifier_layers:
            x = layer(x)

        predictions = x
        loss = predictions[:, class_index]

    # --------------------------------------------------
    # Compute Grad-CAM
    # --------------------------------------------------
    gradients = tape.gradient(loss, conv_output)
    pooled_gradients = tf.reduce_mean(
        gradients, axis=(0, 1, 2)
    )

    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(
        conv_output * pooled_gradients, axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()


def overlay_gradcam(image, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on original image.
    """

    heatmap = cv2.resize(
        heatmap, (image.shape[1], image.shape[0])
    )

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(
        heatmap, cv2.COLORMAP_JET
    )

    return cv2.addWeighted(
        image, 1 - alpha, heatmap, alpha, 0
    )
