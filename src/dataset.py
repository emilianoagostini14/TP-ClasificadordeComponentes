"""
dataset.py - Divide las imágenes capturadas en train (80%) y val (20%)

Uso:
    python src/dataset.py
"""

import os
import shutil
import random

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR     = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR  = os.path.join(BASE_DIR, "data", "processed")

CLASES = [
    "resistencia",
    "led",
    "capacitor_electrolitico",
    "capacitor_ceramico",
]

TRAIN_RATIO = 0.8
RANDOM_SEED = 42
EXTENSIONES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def limpiar_processed():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        print("  Carpeta processed/ limpiada.\n")
    os.makedirs(OUTPUT_DIR)

def dividir_clase(clase: str) -> dict:
    carpeta_origen = os.path.join(RAW_DIR, clase)
    imagenes = [
        f for f in os.listdir(carpeta_origen)
        if f.lower().endswith(EXTENSIONES)
    ]
    if not imagenes:
        print(f"  ⚠  {clase}: no hay imágenes, se omite.")
        return {"train": 0, "val": 0}

    random.seed(RANDOM_SEED)
    random.shuffle(imagenes)

    corte = int(len(imagenes) * TRAIN_RATIO)
    train_imgs = imagenes[:corte]
    val_imgs   = imagenes[corte:]

    for split, lista in [("train", train_imgs), ("val", val_imgs)]:
        destino = os.path.join(OUTPUT_DIR, split, clase)
        os.makedirs(destino, exist_ok=True)
        for img in lista:
            shutil.copy(
                os.path.join(carpeta_origen, img),
                os.path.join(destino, img)
            )

    return {"train": len(train_imgs), "val": len(val_imgs)}

def main():
    print("\n" + "=" * 50)
    print("  PREPARACIÓN DEL DATASET")
    print("=" * 50)

    limpiar_processed()

    total_train = 0
    total_val   = 0

    for clase in CLASES:
        conteo = dividir_clase(clase)
        total_train += conteo["train"]
        total_val   += conteo["val"]
        print(f"  {clase:<30} train: {conteo['train']:>3}  |  val: {conteo['val']:>3}")

    print("=" * 50)
    print(f"  TOTAL                          train: {total_train:>3}  |  val: {total_val:>3}")
    print("=" * 50)
    print("\n  Dataset listo en data/processed/")
    print("  Podés continuar con: python src/train.py\n")

if __name__ == "__main__":
    main()