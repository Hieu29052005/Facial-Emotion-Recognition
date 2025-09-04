import argparse
from utils.data_loader import load_data
from models.resnet_like import build_resnet_like
from models.transfer_model import build_transfer_model
from utils.plot_utils import plot_training
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", choices=["resnet_like", "transfer"], default="resnet_like")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--img_size", type=int, default=224)
args = parser.parse_args()

train_gen, val_gen, num_classes = load_data(
    "data/train", "data/test", img_size=args.img_size, batch_size=args.batch_size
)

input_shape = (args.img_size, args.img_size, 3)

if args.model_type == "resnet_like":
    model = build_resnet_like(input_shape, num_classes)
else:
    model = build_transfer_model(input_shape, num_classes)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs)

model.save("model.h5")
plot_training(history)
