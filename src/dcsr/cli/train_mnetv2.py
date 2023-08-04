import argparse
import logging

import mlflow
import mlflow.keras
import tensorflow as tf
import tensorflow_datasets as tfds
from dcsr.models import mnetv2, prune
from dcsr.models.preprocess_data import cifar10


def train_mnetv2_model(
    training_epochs: int,
    pruning_epochs: int,
    retraining_epochs: int,
    dropout: float,
    width: str,
    sparsity: float,
    skip_blocks: int,
    base_model_path: str,
):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    mlflow.set_experiment(experiment_name="Sparse MobileNetV2")

    (train_ds, val_ds), info = tfds.load(
        name="cifar10", data_dir=".", split=["train[:80%]", "train[80%:]"], as_supervised=True, with_info=True
    )
    # test_ds = tfds.load(name="cifar10", data_dir=".", split="test", as_supervised=True)

    im_size = 96
    num_classes = len(info.features["label"].names)

    batch_size = 128
    train_ds = train_ds.map(
        lambda img, label: (cifar10(img, im_size, augment=True), label), num_parallel_calls=AUTOTUNE
    ).repeat(3)
    val_ds = train_ds.map(lambda img, label: (cifar10(img, im_size), label), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(batch_size).cache().prefetch(AUTOTUNE)
    val_ds = val_ds.batch(batch_size).cache().prefetch(AUTOTUNE)

    # Try loading weights from disk and retrain if they are not present
    try:
        model = tf.keras.models.load_model(base_model_path)
        logging.debug("Using model from {} as base model".format(base_model_path))
    except Exception as e:
        logging.debug("Loading base_model {} failed with {}".format(base_model_path, e))
        logging.debug("Training Dense Base Model from scratch")
        print(width)
        model = mnetv2.model(width, dropout, num_classes, all_trainable=True, im_size=im_size)

        with mlflow.start_run(run_name="Train Base Model"):
            mlflow.log_param("Epochs", training_epochs)
            mlflow.log_param("Batch Size", batch_size)

            history = mnetv2.train(model, train_ds, val_ds, epochs=training_epochs)
            # log_history(history)
            mlflow.keras.log_model(model, "models")

    logging.debug("Pruning to {}% sparsity".format(int(sparsity * 100)))
    with mlflow.start_run(run_name="Pruning"):
        mlflow.log_param("Epochs", pruning_epochs)
        mlflow.log_param("Batch Size", batch_size)
        mlflow.log_param("Sparsity", sparsity)
        mlflow.log_param("Retraining Epochs", retraining_epochs)

        blacklist = ["expanded_conv_project"]
        blacklist += ["block_{}_expand".format(i + 1) for i in range(skip_blocks)]
        blacklist += ["block_{}_project".format(i + 1) for i in range(skip_blocks)]
        logging.debug("Skipping pruning for layers {}".format(blacklist))

        sparse_model, history = prune.prune(
            model,
            train_ds,
            val_ds,
            pruning_epochs=pruning_epochs,
            retraining_epochs=retraining_epochs,
            target_sparsity=float(sparsity),
        )
        # log_history(history)
        mlflow.keras.log_model(sparse_model, "models")


def cli():
    parser = argparse.ArgumentParser(description="Training and Pruning for Keyword Spotting DS-CNN")
    parser.add_argument("-e", "--epochs", type=int, default=60, help="Number of Training Epochs for Dense Base Model")
    parser.add_argument("-p", "--pruning_epochs", type=int, default=40, help="Number of Pruning Epochs")
    parser.add_argument(
        "-r", "--retraining_epochs", type=int, default=20, help="Number of Retraining Epochs after Pruning"
    )
    parser.add_argument(
        "-o",
        "--dropout",
        type=float,
        default=0.1,
        help="Amount of Dropout to add to MobileNetV2 to prevent overfitting",
    )
    parser.add_argument("-w", "--width", type=float, default=1.0, help="Width Multiplier for MobileNetV2")
    parser.add_argument("-s", "--sparsity", type=float, default=0.75, help="Sparsity to prune to")
    parser.add_argument(
        "-x", "--skip_blocks", type=int, default=11, help="Number of initial inverted residual blocks that remain dense"
    )
    parser.add_argument("-b", "--base_model_path", type=str, default="", help="Pre-trained dense base model to prune")
    parser.add_argument(
        "-d",
        "--debug",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
        help="Print Debug Output",
    )
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    train_mnetv2_model(
        args.epochs,
        args.pruning_epochs,
        args.retraining_epochs,
        args.dropout,
        args.width,
        args.sparsity,
        args.skip_blocks,
        args.base_model_path,
    )


if __name__ == "__main__":
    cli()
