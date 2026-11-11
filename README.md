# Handwritten Digit Recognition

[![CI](https://github.com/VishChalla/HandwrittenCharacterNumberIdentification/actions/workflows/ci.yml/badge.svg)](https://github.com/VishChalla/HandwrittenCharacterNumberIdentification/actions/workflows/ci.yml)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-ff6f00)
![License](https://img.shields.io/badge/license-MIT-green)

**A convolutional neural network that reads handwritten digits, with a web app where you upload a photo and see what the model sees.**

Trained on MNIST (70,000 images of the digits 0 to 9). Test accuracy: **98.94%** (106 errors on 10,000 held-out images).

## The interesting part: making it work on real photos

A model that scores 99% on MNIST usually fails on a photo of a digit written on paper. MNIST digits are white on black, scaled to a 20-pixel box and centred. A photo is dark on light, off-centre and any size.

The first version of this app just resized the upload to 28x28. When I revisited the project I measured what that costs, using 300 test digits redrawn the way they would look on paper (dark ink, white background, larger image):

| Image preparation | Correct out of 300 |
|---|---|
| Resize only (original app) | 68 |
| **Detect background, strip frames, crop, scale and centre** | **299** |

Same model, same digits. The preparation step in [`digits/preprocess.py`](digits/preprocess.py) is the difference between a demo that works and one that does not. The app shows the prepared 28x28 image next to the upload so you can see exactly what the model was given.

## Model

Two convolution layers with batch normalisation, max pooling, dropout and a dense head ([`digits/model.py`](digits/model.py)). Random rotation, zoom and shift are built into the model as layers, so they apply only during training.

| | |
|---|---|
| Training data | 54,000 images, plus 6,000 for validation |
| Test data | 10,000 images, used once at the end |
| Epochs | 12, with early stopping and learning-rate reduction on validation loss |
| Test accuracy | 98.94% |
| Hardest digit | 9 (96.4% correct); every other digit is above 98.5% |

The first version used the test set for early stopping, which lets test data influence training. This version splits a validation set off the training data instead, so the test accuracy is an honest estimate.

## Run it

```bash
git clone https://github.com/VishChalla/HandwrittenCharacterNumberIdentification.git
cd HandwrittenCharacterNumberIdentification
pip install -r requirements.txt
```

Start the app (a trained model is included):

```bash
streamlit run app.py
```

Retrain from scratch (about 20 minutes on a laptop CPU):

```bash
python train.py
```

Run the tests (they cover image preparation and need only NumPy and Pillow):

```bash
pytest -q
```

## Project layout

```
app.py                 Streamlit app
train.py               training and evaluation
digits/model.py        CNN architecture
digits/preprocess.py   photo-to-MNIST image preparation
models/                trained model and its metrics
tests/                 8 tests for image preparation
notebooks/             the original exploration notebook
```

## Limitations

- Digits only, one digit per image. It does not read letters or multi-digit numbers.
- The 299-of-300 figure uses MNIST digits re-rendered as if on paper, not real phone photos; shadows, lined paper and pen colour will make real uploads harder.
- Results are from a single training run.

Built by [Vish Challa](https://github.com/VishChalla).
