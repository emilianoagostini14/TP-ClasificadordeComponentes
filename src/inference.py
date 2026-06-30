"""
inference.py - Clasificación de componentes en tiempo real con webcam

Uso:
    python src/inference.py

Controles:
    'q' o ESC → cerrar
"""

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from torchvision import transforms, models
from PIL import Image

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")

CAMARA_INDEX     = 0
IMG_SIZE         = 224
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UMBRAL_CONFIANZA = 0.6

# Cantidad de frames recientes a promediar para suavizar la predicción.
# Más alto = más estable pero reacciona más lento a un cambio de componente.
# Más bajo = reacciona rápido pero oscila más.
BUFFER_FRAMES = 12

COLORES = {
    "resistencia":            (0,   165, 255),  # Naranja
    "led":                    (0,   255, 255),  # Amarillo
    "capacitor_electrolitico":(0,   255, 0),    # Verde
    "capacitor_ceramico":     (255, 0,   255),  # Magenta
}

inferencia_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def cargar_modelo():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No se encontró el modelo en {MODEL_PATH}\n"
            "Asegurate de haber ejecutado primero: python src/train.py"
        )
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    clases     = checkpoint["clases"]
    num_clases = len(clases)

    modelo = models.mobilenet_v2(weights=None)
    in_features = modelo.classifier[1].in_features
    modelo.classifier[1] = nn.Linear(in_features, num_clases)
    modelo.load_state_dict(checkpoint["modelo_state"])
    modelo.to(DEVICE)
    modelo.eval()
    return modelo, clases

def preprocesar_frame(frame):
    imagen_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagen_pil = Image.fromarray(imagen_rgb)
    tensor     = inferencia_transform(imagen_pil)
    return tensor.unsqueeze(0).to(DEVICE)

def predecir_probabilidades(modelo, tensor):
    """
    Corre el modelo sobre un tensor y retorna el vector completo de
    probabilidades (una por clase), sin quedarse solo con la ganadora.
    Esto permite promediarlo con frames anteriores antes de decidir.
    """
    with torch.no_grad():
        salidas        = modelo(tensor)
        probabilidades = torch.softmax(salidas, dim=1)[0]
    return probabilidades.cpu().numpy()


def suavizar_prediccion(buffer_probabilidades, clases):
    """
    Promedia las probabilidades guardadas en el buffer (últimos N frames)
    y devuelve la clase con mayor probabilidad promedio junto a su confianza.
    """
    promedio = np.mean(buffer_probabilidades, axis=0)
    idx = np.argmax(promedio)
    return clases[idx], float(promedio[idx])

def dibujar_overlay(frame, clase, confianza):
    h, w  = frame.shape[:2]
    color = COLORES.get(clase, (255, 255, 255))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    if confianza >= UMBRAL_CONFIANZA:
        texto_clase     = clase.replace("_", " ").upper()
        texto_confianza = f"{confianza:.1%}"
    else:
        texto_clase     = "?"
        texto_confianza = "Confianza baja"
        color           = (128, 128, 128)

    cv2.putText(frame, texto_clase, (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(frame, texto_confianza, (15, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    barra_w = int((w - 30) * min(confianza, 1.0))
    cv2.rectangle(frame, (15, 74), (w - 15, 78), (60, 60, 60), -1)
    cv2.rectangle(frame, (15, 74), (15 + barra_w, 78), color, -1)

    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (255, 255, 255), 1)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (255, 255, 255), 1)

    cv2.putText(frame, "q / ESC = salir", (15, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    return frame

def main():
    print("\n  Cargando modelo...")
    modelo, clases = cargar_modelo()
    print(f"  Modelo cargado. Clases: {clases}")
    print(f"  Dispositivo: {DEVICE}")
    print("  Abriendo cámara...\n")

    cap = cv2.VideoCapture(CAMARA_INDEX)
    if not cap.isOpened():
        print(f"  ERROR: No se pudo abrir la cámara (índice {CAMARA_INDEX}).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("  Cámara abierta. Mostrá un componente frente a la cámara.")
    print("  Presioná 'q' o ESC para salir.\n")

    # Buffer que guarda las últimas N predicciones (vectores de probabilidad)
    # para promediarlas y suavizar la salida en pantalla.
    buffer_probabilidades = deque(maxlen=BUFFER_FRAMES)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("  ERROR: No se pudo leer el frame.")
            break

        tensor = preprocesar_frame(frame)
        probs  = predecir_probabilidades(modelo, tensor)
        buffer_probabilidades.append(probs)

        clase, confianza = suavizar_prediccion(buffer_probabilidades, clases)
        frame             = dibujar_overlay(frame, clase, confianza)

        cv2.imshow("Clasificador de Componentes", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("  Clasificador cerrado.\n")

if __name__ == "__main__":
    main()