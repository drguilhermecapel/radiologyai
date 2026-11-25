import os
import sys
import gc
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


# ───────────────────────────────────────────────────────────
#  Treinador otimizado NIH Chest X-ray
# ───────────────────────────────────────────────────────────
class NIHXrayTrainerOptimized:
    def __init__(self, data_path="D:/NIH_CHEST_XRAY/", batch_size=16):
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.image_size = (380, 380)          # EfficientNetB4
        self.num_classes = 14
        self.start_time = datetime.now()

        self.pathologies = [
            "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
            "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
            "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia"
        ]

        self._setup_logging()
        self._configure_tf()
        self._build_augmenter()
        self.logger.info("Trainer inicializado.")

    # ───────────────────────────────────────────────────────
    #  LOGGING & TF
    # ───────────────────────────────────────────────────────
    def _setup_logging(self):
        os.makedirs("logs", exist_ok=True)
        fname = f"logs/nih_{self.start_time:%Y%m%d_%H%M%S}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[logging.FileHandler(fname, encoding="utf-8"),
                      logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger("NIH")

    def _configure_tf(self):
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            self.logger.info(f"GPU detectada: {gpus[0].name}")
        else:
            self.logger.info("Nenhuma GPU encontrada — rodando em CPU.")

    def _build_augmenter(self):
        self.augment = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.05, fill_mode="reflect"),
                tf.keras.layers.RandomZoom(
                    height_factor=(-0.1, 0.1),
                    width_factor=(-0.1, 0.1),
                    fill_mode="reflect",
                ),
                tf.keras.layers.RandomTranslation(
                    height_factor=0.1,
                    width_factor=0.1,
                    fill_mode="reflect",
                ),
                tf.keras.layers.RandomContrast(0.2),
            ],
            name="aug",
        )

    # ───────────────────────────────────────────────────────
    #  DADOS
    # ───────────────────────────────────────────────────────
    def _find_images(self):
        self.logger.info("Indexando imagens...")
        self.img_paths = {}
        dirs = list(self.data_path.glob("images_*"))
        if not dirs:
            single = self.data_path / "images"
            if single.exists():
                dirs = [single]

        if not dirs:
            self.logger.error("Nenhum diretório de imagens encontrado.")
            return False

        for d in dirs:
            for p in d.rglob("*.png"):
                self.img_paths[p.name] = str(p)
        self.logger.info(f"Imagens indexadas: {len(self.img_paths):,}")
        return True

    def _load_data(self):
        csv = self.data_path / "Data_Entry_2017_v2020.csv"
        if not csv.exists():
            csv = self.data_path / "Data_Entry_2017.csv"
        if not csv.exists():
            self.logger.error("CSV de metadados não encontrado!")
            return False

        self.logger.info(f"Lendo {csv.name}...")
        df = pd.read_csv(csv)

        if not self._find_images():
            return False

        df = df[df["Image Index"].isin(self.img_paths)].reset_index(drop=True)
        df["full_path"] = df["Image Index"].map(self.img_paths)
        for p in self.pathologies:
            df[p] = df["Finding Labels"].str.contains(p).astype(float)

        total = len(df)
        self.logger.info(f"Amostras totais: {total:,}")
        for p in self.pathologies:
            pct = 100 * df[p].sum() / total
            self.logger.info(f"  {p:>18}: {pct:5.2f}%")

        train_val, self.test_df = train_test_split(df, test_size=0.10, random_state=42)
        self.train_df, self.val_df = train_test_split(train_val, test_size=0.1111, random_state=42)
        self.logger.info(
            f"Split -> Treino: {len(self.train_df):,} | Val: {len(self.val_df):,} | "
            f"Teste: {len(self.test_df):,}"
        )
        return True

    # ───────────────────────────────────────────────────────
    #  PIPELINES tf.data
    # ───────────────────────────────────────────────────────
    def _preprocess(self, path, label):
        try:
            if tf.is_tensor(path):
                path = path.numpy().decode()
            img = tf.io.read_file(path)
            img = tf.image.decode_png(img, channels=1)
            img = tf.image.resize(img, self.image_size)
            img = tf.cast(img, tf.float32)          # 0–255
            img = tf.repeat(img, 3, axis=-1)
            return img, label
        except Exception as e:
            self.logger.warning(f"Falha ao processar {path}: {e}")
            return tf.zeros((*self.image_size, 3), tf.float32), label

    def _dataset(self, df, train=True):
        paths = df["full_path"].values
        labels = df[self.pathologies].values.astype(np.float32)

        def gen():
            idx = np.arange(len(paths))
            if train:
                np.random.shuffle(idx)
            for i in idx:
                p = paths[i]
                if os.path.exists(p):
                    img, lab = self._preprocess(p, labels[i])
                    if train:
                        img = self.augment(img, training=True)
                    yield img, lab

        spec = (
            tf.TensorSpec(shape=(*self.image_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(self.num_classes,), dtype=tf.float32),
        )
        ds = tf.data.Dataset.from_generator(gen, output_signature=spec)
        if train:
            ds = ds.repeat()
        return ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    # ───────────────────────────────────────────────────────
    #  MODELO
    # ───────────────────────────────────────────────────────
    def _build_model(self):
        base = tf.keras.applications.EfficientNetB4(
            include_top=False, weights="imagenet", input_shape=(*self.image_size, 3)
        )
        base.trainable = False

        inputs = tf.keras.Input(shape=(*self.image_size, 3))
        x = base(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation="sigmoid")(x)
        return tf.keras.Model(inputs, outputs, name="EffNetB4_NIH")

    def _class_weights(self):
        pos = self.train_df[self.pathologies].sum().values
        total = len(self.train_df)
        return tf.constant([(total - p) / (p or 1) for p in pos], tf.float32)

    def _loss_fn(self):
        w = self._class_weights()

        def loss(y_true, y_pred):
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
            pos = y_true * -tf.math.log(y_pred) * w
            neg = (1 - y_true) * -tf.math.log(1 - y_pred)
            return tf.reduce_mean(pos + neg)

        return loss

    def _compile(self, model, lr, wd=None):
        opt = tf.keras.optimizers.AdamW(lr, weight_decay=wd) if wd else tf.keras.optimizers.Adam(lr)
        model.compile(
            optimizer=opt,
            loss=self._loss_fn(),
            metrics=[tf.keras.metrics.AUC(multi_label=True, num_labels=self.num_classes, name="auc")],
        )
        self.logger.info(f"Compilado com {opt.__class__.__name__} | lr={lr} | wd={wd}")
        return model

    # ───────────────────────────────────────────────────────
    #  TREINAMENTO
    # ───────────────────────────────────────────────────────
    def train(self, warmup=5, epochs=100):
        tr_ds = self._dataset(self.train_df, True)
        va_ds = self._dataset(self.val_df, False)
        steps = len(self.train_df) // self.batch_size
        vsteps = len(self.val_df) // self.batch_size

        model = self._compile(self._build_model(), 1e-3)

        model.fit(
            tr_ds,
            epochs=warmup,
            steps_per_epoch=steps,
            validation_data=va_ds,
            validation_steps=vsteps,
            callbacks=[tf.keras.callbacks.ModelCheckpoint("best_warmup.h5", save_best_only=True,
                                                          monitor="val_auc", mode="max")],
            verbose=1,
        )
        model.load_weights("best_warmup.h5")
        self.logger.info("Warm-up concluído — iniciando fine-tuning.")

        model.layers[1].trainable = True                         # descongela base
        sched = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-4,
            decay_steps=(epochs - warmup) * steps,
            alpha=0.1,
        )
        model = self._compile(model, sched, wd=1e-5)

        model.fit(
            tr_ds,
            initial_epoch=0,
            epochs=epochs,
            steps_per_epoch=steps,
            validation_data=va_ds,
            validation_steps=vsteps,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint("best_final.h5", save_best_only=True,
                                                   monitor="val_auc", mode="max"),
                tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                                 patience=5, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                     patience=2, min_lr=1e-7),
            ],
            verbose=1,
        )
        model.load_weights("best_final.h5")
        return model

    # ───────────────────────────────────────────────────────
    #  AVALIAÇÃO
    # ───────────────────────────────────────────────────────
    def evaluate(self, model):
        te_ds = self._dataset(self.test_df, False)
        preds = model.predict(te_ds, steps=int(np.ceil(len(self.test_df)/self.batch_size)), verbose=1)
        true = self.test_df[self.pathologies].values[: len(preds)]

        res = {}
        for i, p in enumerate(self.pathologies):
            if true[:, i].sum() == 0:
                continue
            auc = roc_auc_score(true[:, i], preds[:, i])
            f1 = f1_score(true[:, i], (preds[:, i] >= 0.5))
            res[p] = {"AUC": auc, "F1": f1}
            self.logger.info(f"{p:>18}: AUC={auc:.4f} | F1={f1:.4f}")

        mean_auc = np.mean([r["AUC"] for r in res.values()])
        mean_f1 = np.mean([r["F1"] for r in res.values()])
        self.logger.info(f"==> AUC médio: {mean_auc:.4f} | F1 médio: {mean_f1:.4f}")

        with open("evaluation_results.json", "w") as f:
            json.dump({"mean_auc": float(mean_auc), "mean_f1": float(mean_f1),
                       "per_class": {k: {m: float(v) for m, v in d.items()} for k, d in res.items()}},
                      f, indent=2)

    # ───────────────────────────────────────────────────────
    #  EXECUÇÃO COMPLETA
    # ───────────────────────────────────────────────────────
    def run(self):
        if not self._load_data():
            return
        model = self.train(warmup=5, epochs=100)
        self.evaluate(model)
        self.logger.info("Pipeline finalizado.")


# ───────────────────────────────────────────────────────────
#  MAIN
# ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 80)
    print("NIH CHEST X-RAY – VERSÃO 100 ÉPOCAS")
    print("=" * 80)
    trainer = NIHXrayTrainerOptimized(
        data_path="D:/NIH_CHEST_XRAY/",   # ajuste se necessário
        batch_size=16                     # aumente se tiver GPU com VRAM
    )
    trainer.run()
