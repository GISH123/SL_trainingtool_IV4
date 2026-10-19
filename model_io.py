from __future__ import annotations

import json
import shutil
from pathlib import Path

import configuration as cfg


def export_saved_model(model, export_dir: Path) -> None:
    import tensorflow as tf

    export_dir = Path(export_dir)
    temp_dir = export_dir.with_name(export_dir.name + ".tmp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    class ServingModule(tf.Module):
        def __init__(self, keras_model):
            super().__init__()
            self.model = keras_model

        @tf.function(input_signature=[tf.TensorSpec(
            [None, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, cfg.CHANNELS],
            tf.float32, name="images"
        )])
        def serving_default(self, images):
            return {"logits": self.model(images, training=False)}

    module = ServingModule(model)
    tf.saved_model.save(
        module, str(temp_dir),
        signatures={"serving_default": module.serving_default},
    )
    if export_dir.exists():
        shutil.rmtree(export_dir)
    temp_dir.replace(export_dir)


def save_training_state(epoch: int, logs: dict | None = None) -> None:
    cfg.TRAINING_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"last_completed_epoch": int(epoch), "logs": logs or {}}
    cfg.TRAINING_STATE_PATH.write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8"
    )


def load_training_state() -> dict:
    if not cfg.TRAINING_STATE_PATH.exists():
        return {"last_completed_epoch": 0, "logs": {}}
    return json.loads(cfg.TRAINING_STATE_PATH.read_text(encoding="utf-8"))


class SavedModelCallable:
    def __init__(self, model_dir: Path):
        import tensorflow as tf

        self.module = tf.saved_model.load(str(model_dir))
        signatures = getattr(self.module, "signatures", {})
        self.fn = signatures.get("serving_default")
        if self.fn is None and callable(self.module):
            self.fn = self.module
        if self.fn is None:
            raise RuntimeError(f"No callable serving signature in {model_dir}")

    def __call__(self, batch):
        output = self.fn(batch)
        if isinstance(output, dict):
            if "logits" in output:
                return output["logits"]
            return next(iter(output.values()))
        return output


def load_saved_model_callable(model_dir: Path = cfg.LATEST_MODEL_DIR) -> SavedModelCallable:
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"SavedModel not found: {model_dir}")
    return SavedModelCallable(model_dir)
