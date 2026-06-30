"""
manual_crop.py - Herramienta interactiva para recortar manualmente las fotos
que el recorte automático (crop_components.py) no logró procesar bien.

Cómo funciona:
    - Te muestra cada foto de data/raw/<clase>/, una por una
    - Arrastrás el mouse para dibujar un rectángulo alrededor del componente
    - ENTER o ESPACIO  -> confirma el recorte y guarda
    - 'r'              -> reinicia el rectángulo en la foto actual
    - 's'              -> salta la foto sin modificarla (se copia tal cual)
    - 'q' o ESC         -> termina la sesión (lo ya hecho queda guardado)

"""

import os
import cv2
import shutil

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR     = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR  = os.path.join(BASE_DIR, "data", "raw_cropped_manual")

CLASES = [
    "resistencia",
    "led",
    "capacitor_electrolitico",
    "capacitor_ceramico",
]

EXTENSIONES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Tamaño máximo de ventana para mostrar (si la foto es más grande, se escala
# solo para mostrarla; el recorte final se hace sobre la resolución original)
VENTANA_MAX = 900

# ─────────────────────────────────────────────
# ESTADO GLOBAL DEL MOUSE (necesario para el callback de OpenCV)
# ─────────────────────────────────────────────

estado = {
    "dibujando": False,
    "punto_inicio": None,
    "punto_actual": None,
    "rectangulo_final": None,
}


def callback_mouse(event, x, y, flags, param):
    """Maneja el arrastre del mouse para dibujar el rectángulo de recorte."""
    if event == cv2.EVENT_LBUTTONDOWN:
        estado["dibujando"] = True
        estado["punto_inicio"] = (x, y)
        estado["rectangulo_final"] = None

    elif event == cv2.EVENT_MOUSEMOVE:
        if estado["dibujando"]:
            estado["punto_actual"] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        estado["dibujando"] = False
        estado["punto_actual"] = (x, y)
        estado["rectangulo_final"] = (estado["punto_inicio"], (x, y))


def normalizar_rectangulo(p1, p2):
    """Ordena las coordenadas para que (x1,y1) sea la esquina superior izquierda."""
    x1, y1 = p1
    x2, y2 = p2
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def procesar_imagen(ruta_imagen: str, nombre_ventana: str):
    """
    Muestra una imagen, permite dibujar el rectángulo de recorte,
    y retorna la imagen recortada (o None si se decide saltar).
    """
    imagen_original = cv2.imread(ruta_imagen)
    if imagen_original is None:
        return None, "error"

    alto_orig, ancho_orig = imagen_original.shape[:2]

    # Calcular escala para mostrar en pantalla sin que sea gigante
    escala = min(VENTANA_MAX / ancho_orig, VENTANA_MAX / alto_orig, 1.0)
    imagen_mostrar = cv2.resize(imagen_original, None, fx=escala, fy=escala)

    estado["rectangulo_final"] = None
    estado["dibujando"] = False

    cv2.namedWindow(nombre_ventana)
    cv2.setMouseCallback(nombre_ventana, callback_mouse)

    while True:
        frame_actual = imagen_mostrar.copy()

        # Dibujar el rectángulo mientras se arrastra el mouse
        if estado["dibujando"] and estado["punto_inicio"] and estado["punto_actual"]:
            cv2.rectangle(frame_actual, estado["punto_inicio"], estado["punto_actual"],
                         (0, 255, 0), 2)

        # Dibujar el rectángulo ya confirmado (antes de apretar ENTER)
        elif estado["rectangulo_final"]:
            p1, p2 = estado["rectangulo_final"]
            cv2.rectangle(frame_actual, p1, p2, (0, 255, 0), 2)

        # Instrucciones en pantalla
        cv2.putText(frame_actual, "Arrastra para recortar | ENTER=guardar | r=reset | s=saltar | q=salir",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow(nombre_ventana, frame_actual)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 32):  # ENTER o ESPACIO -> confirmar
            if estado["rectangulo_final"]:
                p1, p2 = estado["rectangulo_final"]
                x1, y1, x2, y2 = normalizar_rectangulo(p1, p2)

                # Volver a escala original antes de recortar
                x1, y1, x2, y2 = [int(v / escala) for v in (x1, y1, x2, y2)]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(ancho_orig, x2), min(alto_orig, y2)

                if x2 > x1 and y2 > y1:
                    recorte = imagen_original[y1:y2, x1:x2]
                    return recorte, "recortada"
            # Si no hay rectángulo dibujado, no hace nada (sigue esperando)

        elif key == ord('r'):  # reset
            estado["rectangulo_final"] = None

        elif key == ord('s'):  # saltar sin tocar
            return imagen_original, "saltada"

        elif key in (ord('q'), 27):  # salir de toda la sesión
            return None, "salir"


def main():
    print("\n" + "=" * 55)
    print("  RECORTE MANUAL DE FOTOS")
    print("=" * 55)
    print("  Controles:")
    print("    Arrastrar mouse -> dibujar rectángulo")
    print("    ENTER/ESPACIO   -> confirmar y guardar")
    print("    r               -> reiniciar rectángulo")
    print("    s               -> saltar foto (se copia sin recortar)")
    print("    q / ESC         -> terminar sesión")
    print("=" * 55 + "\n")

    nombre_ventana = "Recorte manual"

    for clase in CLASES:
        carpeta_origen  = os.path.join(RAW_DIR, clase)
        carpeta_destino = os.path.join(OUTPUT_DIR, clase)
        os.makedirs(carpeta_destino, exist_ok=True)

        archivos = [
            f for f in os.listdir(carpeta_origen)
            if f.lower().endswith(EXTENSIONES)
        ]

        print(f"\n  Clase: {clase} ({len(archivos)} imágenes)")

        for i, nombre_archivo in enumerate(archivos, start=1):
            ruta_destino = os.path.join(carpeta_destino, nombre_archivo)

            # Si ya fue procesada en una sesión anterior, la salta automáticamente
            if os.path.exists(ruta_destino):
                continue

            ruta_origen = os.path.join(carpeta_origen, nombre_archivo)
            titulo = f"{nombre_ventana} - {clase} [{i}/{len(archivos)}] {nombre_archivo}"

            resultado, accion = procesar_imagen(ruta_origen, nombre_ventana)
            cv2.setWindowTitle(nombre_ventana, titulo) if resultado is not None else None

            if accion == "salir":
                cv2.destroyAllWindows()
                print("\n  Sesión terminada. Lo ya procesado quedó guardado.")
                print(f"  Resultado en: {OUTPUT_DIR}\n")
                return

            elif accion == "error":
                print(f"    ⚠  No se pudo leer: {nombre_archivo}")
                continue

            cv2.imwrite(ruta_destino, resultado)
            etiqueta = "✓ recortada" if accion == "recortada" else "→ saltada (sin cambios)"
            print(f"    [{i}/{len(archivos)}] {nombre_archivo}  {etiqueta}")

    cv2.destroyAllWindows()
    print("\n  Todas las fotos fueron procesadas.")
    print(f"  Resultado en: {OUTPUT_DIR}")
    print("\n  Si te convence el resultado, fusionalo con tus fotos ya recortadas")
    print("  copiando manualmente los archivos a data/raw/ o data/raw_cropped/\n")


if __name__ == "__main__":
    main()