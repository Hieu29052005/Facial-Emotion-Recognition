# Facial Emotion Recognition (FER)

This project is a complete pipeline for **Facial Emotion Recognition** (or general image classification). It includes:

* **Custom ResNet-like model**
* **Transfer Learning (ResNet50, EfficientNet, etc.)**
* **Training pipeline**
* **Prediction via CLI or real-time webcam**
* **Web app with Streamlit for deployment**

---

## 🚀 Project Structure

```
image_classifier/
│
├── data/                     # Dataset (train/test folders)
│
├── models/                   
│   ├── resnet_like.py        # Custom ResNet-like CNN
│   ├── transfer_model.py     # Transfer learning models
│
├── utils/
│   ├── data_loader.py        # Data loading & preprocessing
│   ├── plot_utils.py         # Training plots, confusion matrix, report
│
├── train.py                  # Train model (custom or transfer)
├── predict.py                # Predict from image or webcam
├── app.py                    # Streamlit web app
│
├── requirements.txt
└── README.md
```

---

## 📦 Installation

```bash
# Clone project
https://github.com/yourusername/image_classifier.git
cd image_classifier

# Install dependencies
pip install -r requirements.txt
```

---

## 📂 Dataset Setup

Organize dataset in this format:

```
data/
├── train/
│   ├── happy/
│   ├── sad/
│   ├── angry/
│   └── ...
├── test/
    ├── happy/
    ├── sad/
    ├── angry/
    └── ...
```

---

## 🏋️ Training

Train a **ResNet-like custom model**:

```bash
python train.py --model_type resnet_like --epochs 20 --batch_size 64 --img_size 224
```

Train a **transfer learning model**:

```bash
python train.py --model_type transfer --epochs 20 --batch_size 64 --img_size 224
```

The trained model will be saved as `model.h5`.

---

## 🔍 Prediction

### Single image

```bash
python predict.py --model model.h5 --image path/to/image.jpg --classes angry happy sad surprise neutral fear disgust
```

### Real-time (webcam)

```bash
python predict.py --model model.h5 --classes angry happy sad surprise neutral fear disgust
```

Press **q** to quit.

---

## 🌐 Web App (Streamlit)

```bash
streamlit run app.py
```

* Upload an image → see predicted class and confidence scores.

---

## 📊 Evaluation

Use `plot_utils.py` to:

* Plot accuracy & loss curves
* Show confusion matrix
* Print classification report

Example usage inside training script:

```python
from utils.plot_utils import plot_confusion_report

# After training
preds = model.predict(val_gen)
y_true = val_gen.classes
y_pred = preds.argmax(axis=1)
class_names = list(val_gen.class_indices.keys())
plot_confusion_report(y_true, y_pred, class_names)
```

---

## ⚙️ Requirements

* Python 3.8+
* TensorFlow 2.x
* Keras
* OpenCV
* Streamlit
* scikit-learn
* matplotlib

Install all with:

```bash
pip install -r requirements.txt
```

---

## 📌 Deployment

You can deploy on:

* **Streamlit Cloud**
* **Hugging Face Spaces**
* **Render**
* **Docker**

---

## ✨ Author

Nguyễn Vương Trung Hiếu
