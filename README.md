# Garbage Classification with CNNs (TensorFlow / Keras)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-ff6f00)
![Keras](https://img.shields.io/badge/Keras-CNN%20Training-d00000)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Image%20Classification-brightgreen)

## Project Overview

End-to-end **garbage image classification** pipeline implemented in **TensorFlow/Keras**. The project trains multiple lightweight CNN architectures to classify waste images into **five categories**:

- `biological`
- `glass`
- `metal`
- `paper`
- `plastic`

The repository includes dataset loading from a directory structure, preprocessing/normalization, training with checkpoints, epoch-time profiling, and post-training evaluation with detailed classification metrics.

---

## Features / Capabilities

- **TensorFlow/Keras training pipeline**
  - Uses `tf.keras.utils.image_dataset_from_directory(...)` for efficient dataset loading and batching.
  - CNN models built with `Sequential`, `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, and optional `Dropout`.

- **Multiple CNN variants**
  - `create_first_cnn_model()` – deeper CNN stack (Conv/Pool blocks + dense head).
  - `create_second_cnn_model()` – smaller CNN + dropout regularization.
  - `create_third_cnn_model()` – 3×3 kernel variant with a larger dense layer.

- **Train/validation/test split experiments**
  - Split logic built directly on `tf.data.Dataset` (`take`, `skip`, `concatenate`).
  - Alternative split strategies (including uniform class-size approach).

- **Checkpoints & model export**
  - Best checkpoints saved automatically via `ModelCheckpoint(..., monitor='val_accuracy', save_best_only=True)`.
  - Trained models can be exported to `.h5`.

- **Performance profiling**
  - Custom `TimeHistory` callback tracks **epoch execution time**.
  - Built-in utilities to compare training curves and time-per-epoch across runs.

- **Evaluation & reporting**
  - Predictions generated over a dataset split and converted to class IDs.
  - Metrics reported using `sklearn.metrics.classification_report`.
  - Per-class **precision / recall / F1** plotted and saved.

- **Dataset diagnostics**
  - Class distribution counting.
  - Image size inspection (min/max and unique sizes).
  - RGB histogram generation per class (OpenCV + Matplotlib).

---

## Tech Stack

- **Language**: Python
- **Deep Learning**: TensorFlow, Keras
- **Computer Vision / I/O**: OpenCV (`cv2`)
- **Metrics**: scikit-learn (`classification_report`, precision/recall/F1)
- **Visualization**: Matplotlib
- **Utilities**: NumPy, Colorama (CLI coloring)
