from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

def build_transfer_model(input_shape, num_classes, base_model_name="ResNet50"):
    if base_model_name == "ResNet50":
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Only ResNet50 is supported right now.")

    base_model.trainable = False  # Freeze base model

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs)
