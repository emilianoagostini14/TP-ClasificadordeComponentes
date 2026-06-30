"""
train.py - Entrenamiento con transfer learning sobre MobileNetV2

"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR  = os.path.join(BASE_DIR, "models")

NUM_CLASES  = 4
EPOCHS      = 20
BATCH_SIZE  = 16
LR          = 0.001
IMG_SIZE    = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def cargar_datasets():
    train_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, "train"),
        transform=train_transforms
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, "val"),
        transform=val_transforms
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader, train_dataset.classes

def construir_modelo(num_clases: int):
    modelo = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for param in modelo.features.parameters():
        param.requires_grad = False
    in_features = modelo.classifier[1].in_features
    modelo.classifier[1] = nn.Linear(in_features, num_clases)
    return modelo.to(DEVICE)

def entrenar_epoca(modelo, loader, criterio, optimizador):
    modelo.train()
    perdida_total = 0
    correctas = 0
    total = 0
    for imagenes, etiquetas in loader:
        imagenes, etiquetas = imagenes.to(DEVICE), etiquetas.to(DEVICE)
        optimizador.zero_grad()
        salidas = modelo(imagenes)
        perdida = criterio(salidas, etiquetas)
        perdida.backward()
        optimizador.step()
        perdida_total += perdida.item()
        _, predicciones = torch.max(salidas, 1)
        correctas += (predicciones == etiquetas).sum().item()
        total     += etiquetas.size(0)
    return perdida_total / len(loader), correctas / total

def evaluar(modelo, loader, criterio):
    modelo.eval()
    perdida_total = 0
    correctas = 0
    total = 0
    with torch.no_grad():
        for imagenes, etiquetas in loader:
            imagenes, etiquetas = imagenes.to(DEVICE), etiquetas.to(DEVICE)
            salidas = modelo(imagenes)
            perdida = criterio(salidas, etiquetas)
            perdida_total += perdida.item()
            _, predicciones = torch.max(salidas, 1)
            correctas += (predicciones == etiquetas).sum().item()
            total     += etiquetas.size(0)
    return perdida_total / len(loader), correctas / total

def main():
    print("\n" + "=" * 50)
    print("  ENTRENAMIENTO - Clasificador de Componentes")
    print("=" * 50)
    print(f"  Dispositivo: {DEVICE}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    train_loader, val_loader, clases = cargar_datasets()
    print(f"  Clases detectadas: {clases}")
    print(f"  Imágenes train: {len(train_loader.dataset)}")
    print(f"  Imágenes val:   {len(val_loader.dataset)}")
    print("=" * 50 + "\n")

    modelo = construir_modelo(NUM_CLASES)
    criterio    = nn.CrossEntropyLoss()
    optimizador = optim.Adam(
        filter(lambda p: p.requires_grad, modelo.parameters()), lr=LR
    )

    mejor_val_acc = 0.0
    ruta_modelo   = os.path.join(MODELS_DIR, "best_model.pth")

    for epoca in range(1, EPOCHS + 1):
        train_loss, train_acc = entrenar_epoca(modelo, train_loader, criterio, optimizador)
        val_loss,   val_acc   = evaluar(modelo, val_loader, criterio)

        print(f"  Época {epoca:>2}/{EPOCHS}  |  "
              f"Train loss: {train_loss:.4f}  acc: {train_acc:.2%}  |  "
              f"Val loss: {val_loss:.4f}  acc: {val_acc:.2%}", end="")

        if val_acc > mejor_val_acc:
            mejor_val_acc = val_acc
            torch.save({
                "modelo_state": modelo.state_dict(),
                "clases":       clases,
                "img_size":     IMG_SIZE,
            }, ruta_modelo)
            print("  ✓ Modelo guardado")
        else:
            print()

    print(f"\n  Entrenamiento finalizado.")
    print(f"  Mejor accuracy en validación: {mejor_val_acc:.2%}")
    print(f"  Modelo guardado en: {ruta_modelo}\n")

if __name__ == "__main__":
    main()