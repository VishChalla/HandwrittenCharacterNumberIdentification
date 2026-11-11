"""Train the digit classifier and report held-out accuracy.

    python train.py                 # full run, with augmentation
    python train.py --epochs 3      # quick run
    python train.py --no-augment    # ablation: same model without augmentation

The 60,000 MNIST training images are split 54,000 / 6,000 into train and
validation. Early stopping and learning-rate decisions use validation only;
the 10,000 test images are touched once, at the end.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import set_random_seed

from digits.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args()
    set_random_seed(0)

    (x, y), (x_test, y_test) = mnist.load_data()
    x = x[..., None].astype("float32") / 255
    x_test = x_test[..., None].astype("float32") / 255
    order = np.random.default_rng(0).permutation(len(x))
    val, train = order[:6000], order[6000:]

    model = build_model(augment=not args.no_augment)
    history = model.fit(
        x[train], y[train], validation_data=(x[val], y[val]), batch_size=64, epochs=args.epochs,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-5),
        ],
        verbose=2,
    )

    _, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x_test, verbose=0).argmax(axis=1)
    per_digit = {str(d): float((predictions[y_test == d] == d).mean()) for d in range(10)}

    args.out.mkdir(exist_ok=True)
    suffix = "_no_augment" if args.no_augment else ""
    model.save(args.out / f"mnist_cnn{suffix}.keras")
    metrics = {
        "augmentation": not args.no_augment,
        "epochs_run": len(history.history["loss"]),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "test_accuracy": float(test_accuracy),
        "test_errors": int((predictions != y_test).sum()),
        "test_accuracy_per_digit": per_digit,
    }
    (args.out / f"metrics{suffix}.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
