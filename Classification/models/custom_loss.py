from tensorflow import keras

def label_smoothing_loss(y_true, y_pred, smoothing=0.1):
    """
    Custom loss function implementing label smoothing.
    """
    num_classes = y_true.shape[-1]
    smooth_positives = 1.0 - smoothing
    smooth_negatives = smoothing / num_classes
    y_true = y_true * smooth_positives + smooth_negatives

    return keras.losses.categorical_crossentropy(y_true, y_pred)
