import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2


def preprocess_frame(frame, target_size=(224,224)):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img_array = img.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def realtime_predict(model_path, class_names):
    model = tf.keras.models.load_model(model_path)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_array = preprocess_frame(frame)
        preds = model.predict(img_array)[0]
        idx = np.argmax(preds)
        label = f"{class_names[idx]} ({preds[idx]*100:.2f}%)"

        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Real-time Prediction', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model.h5', help='Path to trained model')
    parser.add_argument('--classes', type=str, nargs='+', default=None, help='List of class names')
    args = parser.parse_args()

    if args.classes is None:
        args.classes = [f'class_{i}' for i in range(7)]

    realtime_predict(args.model, args.classes)
