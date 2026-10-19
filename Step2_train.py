from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

import configuration as cfg
from class_mapping import load_class_names
from dataset_utils import (
    collect_hard_pool_samples,
    collect_split_samples,
    dataset_summary,
    make_dataset,
    oversample,
)
from image_utils import load_tf_image
from model_io import export_saved_model, load_training_state, save_training_state
from models.inception_v4 import build_inception_v4


class ManualSaveListener:
    def __init__(self):
        self.event = threading.Event()
        self.listener = None

    def start(self) -> None:
        if not cfg.ENABLE_F12_MANUAL_SAVE:
            return
        try:
            from pynput import keyboard

            def on_press(key):
                if key == keyboard.Key.f12:
                    self.event.set()
                    print("[F12] Manual save requested; saving at the next batch boundary.")

            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.daemon = True
            self.listener.start()
        except Exception as exc:
            print(f"[F12] Listener disabled: {exc}")

    def consume(self) -> bool:
        if self.event.is_set():
            self.event.clear()
            return True
        return False


class SaveCallback:
    @staticmethod
    def build(manual_listener: ManualSaveListener):
        import tensorflow as tf

        class _SaveCallback(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.current_epoch = 0

            def on_epoch_begin(self, epoch, logs=None):
                self.current_epoch = int(epoch)

            def _save(self, epoch: int, logs: dict | None, target_dir: Path | None = None):
                cfg.LATEST_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                self.model.save_weights(str(cfg.LATEST_WEIGHTS_PATH))
                export_saved_model(self.model, target_dir or cfg.LATEST_MODEL_DIR)
                save_training_state(epoch, logs)

            def on_train_batch_end(self, batch, logs=None):
                if manual_listener.consume():
                    # The current epoch is not complete yet. Saving this value makes
                    # the next run repeat the current epoch instead of skipping ahead.
                    self._save(self.current_epoch, logs, cfg.LATEST_MODEL_DIR)
                    print("[F12] Manual save completed.")

            def on_epoch_end(self, epoch, logs=None):
                completed = epoch + 1
                if cfg.SAVE_LATEST_EVERY_EPOCH:
                    self._save(completed, logs, cfg.LATEST_MODEL_DIR)
                if cfg.SNAPSHOT_EVERY_EPOCHS > 0 and completed % cfg.SNAPSHOT_EVERY_EPOCHS == 0:
                    snapshot = cfg.SNAPSHOT_DIR / f"epoch_{completed:04d}"
                    snapshot.mkdir(parents=True, exist_ok=True)
                    self.model.save_weights(str(snapshot / "model.weights.h5"))
                    export_saved_model(self.model, snapshot / "saved_model")
                    (snapshot / "logs.json").write_text(
                        json.dumps(logs or {}, indent=2, default=float), encoding="utf-8"
                    )

        return _SaveCallback()


class WorseValidationCallback:
    @staticmethod
    def build(valid_rows, class_names):
        import tensorflow as tf

        class _WorseValidationCallback(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.best = float("-inf")

            def on_epoch_end(self, epoch, logs=None):
                if not cfg.DUMP_WHEN_VALID_WORSE:
                    return
                logs = logs or {}
                value = logs.get("val_accuracy")
                if value is None:
                    value = logs.get("val_sparse_categorical_accuracy")
                if value is None:
                    return
                current = float(value)
                if current >= self.best:
                    self.best = current
                    return

                output = cfg.INSPECT_WORSE_VALID_DIR / f"epoch_{epoch + 1:04d}_val_{current:.6f}"
                if output.exists():
                    shutil.rmtree(output)
                copied = 0
                for image_path, true_index in valid_rows:
                    image = load_tf_image(image_path, augment=False)
                    logits = self.model(tf.expand_dims(image, 0), training=False)
                    pred_index = int(tf.argmax(logits[0]).numpy())
                    if pred_index == true_index:
                        continue
                    true_name = class_names[true_index]
                    pred_name = class_names[pred_index]
                    destination = output / true_name / f"pred_{pred_name}" / Path(image_path).name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(image_path, destination)
                    copied += 1
                    if copied >= cfg.MAX_WORSE_VALID_IMAGES:
                        break
                print(f"[valid worse] copied {copied} mistakes to {output}")

        return _WorseValidationCallback()


def main() -> None:
    import tensorflow as tf

    tf.random.set_seed(cfg.RANDOM_SEED)
    class_names = load_class_names(cfg.CLASS_NAMES_PATH)
    train_rows = collect_split_samples(cfg.TRAIN_DIR, class_names)
    hard_rows = collect_hard_pool_samples(class_names)
    train_rows.extend(hard_rows)
    train_rows = oversample(train_rows, cfg.RANDOM_SEED)
    valid_rows = collect_split_samples(cfg.VALID_DIR, class_names)

    print("class_names:", class_names)
    print("num_classes:", len(class_names))
    print("train summary:", dataset_summary(train_rows))
    print("valid summary:", dataset_summary(valid_rows))
    print("hard_pool accepted:", len(hard_rows))

    train_ds = make_dataset(train_rows, training=True)
    valid_ds = make_dataset(valid_rows, training=False)

    model = build_inception_v4(
        num_classes=len(class_names),
        input_shape=(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, cfg.CHANNELS),
        dropout_keep_prob=cfg.DROPOUT_KEEP_PROB,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    initial_epoch = 0
    if cfg.RESUME_LATEST and cfg.LATEST_WEIGHTS_PATH.exists():
        model.load_weights(str(cfg.LATEST_WEIGHTS_PATH))
        initial_epoch = int(load_training_state().get("last_completed_epoch", 0))
        print(f"Resumed latest weights at epoch {initial_epoch}")
    elif cfg.LOAD_PRETRAINED and cfg.PRETRAINED_WEIGHTS_PATH.exists():
        model.load_weights(str(cfg.PRETRAINED_WEIGHTS_PATH), by_name=True, skip_mismatch=True)
        print(f"Loaded pretrained weights: {cfg.PRETRAINED_WEIGHTS_PATH}")

    manual = ManualSaveListener()
    manual.start()
    callbacks = [
        SaveCallback.build(manual),
        WorseValidationCallback.build(valid_rows, class_names),
        tf.keras.callbacks.CSVLogger(str(cfg.PROJECT_ROOT / "training_history.csv"), append=initial_epoch > 0),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=cfg.EPOCHS,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    main()
