# ============================================================
# DETECTOR DE GESTOS FACIALES + MANOS
# ============================================================
# Reconoce dos gestos en tiempo real usando MediaPipe:
#   - "Topo Gigio" (festejo de Riquelme): ambas manos en las orejas
#   - "Dybala": dedo índice en la boca
# Muestra la cámara con detección facial (468 puntos) y de manos (21 puntos),
# un emoji del gesto y la foto del jugador correspondiente.
# ============================================================

import cv2               # Cámara y procesamiento de imágenes
import mediapipe as mp   # Face Mesh y Hand Landmarks
import os                # Para verificar si existen fotos
import sys               # Para salir
import math              # Para calcular distancias entre puntos
import numpy as np       # Para convertir imágenes PIL a OpenCV
from PIL import Image, ImageDraw, ImageFont  # Para renderizar emojis

# ── DETECCIÓN AUTOMÁTICA DE VERSIÓN DE MEDIAPIPE ────────────
MEDIAPIPE_NUEVO = not hasattr(mp, "solutions")

if MEDIAPIPE_NUEVO:
    import urllib.request
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.vision import FaceLandmarkerOptions, HandLandmarkerOptions, RunningMode

    # ── MODELO DE FACE MESH ─────────────────────────────────
    MODEL_PATH = "face_landmarker.task"
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    if not os.path.exists(MODEL_PATH):
        print("Descargando modelo Face Landmarker...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("OK")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    face_opts = FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_faces=1
    )
    face_mesh = mp_vision.FaceLandmarker.create_from_options(face_opts)

    # ── MODELO DE MANOS ─────────────────────────────────────
    HAND_MODEL_PATH = "hand_landmarker.task"
    HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    if not os.path.exists(HAND_MODEL_PATH):
        print("Descargando modelo Hand Landmarker...")
        try:
            urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
            print("OK")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    hand_opts = HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2  # Detecta hasta 2 manos
    )
    hands = mp_vision.HandLandmarker.create_from_options(hand_opts)
    _ts_mp = 0  # Timestamp para modo VIDEO

else:
    # Versión antigua de MediaPipe (< 0.10)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2)

# ── CONEXIONES DE LA MANO ──────────────────────────────────
# Pares de índices que forman el esqueleto de la mano (según MediaPipe)
CONEXIONES_MANO = [
    (0,1),(1,2),(2,3),(3,4),        # Pulgar
    (0,5),(5,6),(6,7),(7,8),        # Índice
    (0,9),(9,10),(10,11),(11,12),   # Medio
    (0,13),(13,14),(14,15),(15,16), # Anular
    (0,17),(17,18),(18,19),(19,20), # Meñique
]

# ── EMOJIS ─────────────────────────────────────────────────
# Usa la tipografía Apple Color Emoji de macOS para renderizar emojis
# sobre el video. Pillow crea la imagen RGBA y la convertimos a OpenCV.
EMOJI_FONT = "/System/Library/Fonts/Apple Color Emoji.ttc"

def render_emoji(emoji_char, size=80):
    """Renderiza un emoji como imagen OpenCV (BGRA) usando Pillow."""
    try:
        font = ImageFont.truetype(EMOJI_FONT, size)
    except:
        font = ImageFont.load_default()  # Fallback si no encuentra la fuente
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        bbox = draw.textbbox((0, 0), emoji_char, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except:
        tw, th = size // 2, size // 2
    draw.text(((size - tw) // 2, (size - th) // 2), emoji_char, font=font)
    # PIL: RGBA → OpenCV: BGRA
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)

# Pre-renderiza los emojis de cada gesto
EMOJIS = {
    "topo_gigio": render_emoji("🐭"),  # Ratón (Topo Gigio)
    "dybala":     render_emoji("⭐"),  # Estrella (Dybala = "La Joya")
}

# ── FOTO DEL JUGADOR ───────────────────────────────────────
def overlay_foto_lado(frame, nombre_base, max_w=550, max_h=550):
    """
    Busca una foto (jpg/png/jpeg) en assets/ o en la raíz y la superpone
    en el margen derecho del video, con tamaño máximo max_w × max_h.
    Si la imagen tiene canal alpha, hace una mezcla transparente.
    """
    for ext in [".jpg", ".png", ".jpeg"]:
        for prefijo in ["assets/", ""]:
            ruta = prefijo + nombre_base + ext
            if os.path.exists(ruta):
                overlay = cv2.imread(ruta, cv2.IMREAD_UNCHANGED)
                if overlay is None:
                    overlay = cv2.imread(ruta)
                if overlay is not None:
                    oh, ow = overlay.shape[:2]
                    escala = min(max_w / ow, max_h / oh)
                    nw, nh = int(ow * escala), int(oh * escala)
                    overlay = cv2.resize(overlay, (nw, nh))
                    h, w = frame.shape[:2]
                    x1, y1 = w - nw - 20, (h - nh) // 2  # Derecha, centrado vertical
                    if overlay.shape[2] == 4:
                        alfa = overlay[:, :, 3] / 255.0
                        for c in range(3):
                            frame[y1:y1+nh, x1:x1+nw, c] = (
                                frame[y1:y1+nh, x1:x1+nw, c] * (1 - alfa) + overlay[:, :, c] * alfa
                            )
                    else:
                        frame[y1:y1+nh, x1:x1+nw] = overlay
                    return True
    return False  # No se encontró la foto

def dist(p1, p2):
    """Distancia euclideana entre dos landmarks normalizados."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

# ── CÁMARA ──────────────────────────────────────────────────
cap = None
for idx in [0, 1, 2]:
    temp_cap = cv2.VideoCapture(idx)
    if temp_cap.isOpened():
        success, frame_test = temp_cap.read()
        # Verifica que el frame sea válido (no negro)
        if success and frame_test is not None and frame_test.any():
            cap = temp_cap
            print(f"Cámara en índice {idx}")
            break
        temp_cap.release()

if cap is None:
    print("ERROR: No hay cámara.")
    sys.exit(1)

# ── CONFIGURACIÓN DE GESTOS ─────────────────────────────────
# Cada gesto tiene: emoji, nombre, foto asociada y color de depuración
GESTOS = {
    "topo_gigio": {"emoji": "🐭", "label": "TOPO GIGIO!",  "foto": "riquelme", "color": (0, 255, 255)},
    "dybala":     {"emoji": "⭐", "label": "DYBALA!",       "foto": "dybala",   "color": (255, 0, 255)},
}

# ── BUCLE PRINCIPAL ─────────────────────────────────────────
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)                               # Espejo (efecto selfie)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)       # OpenCV BGR → RGB
    h_img, w_img, _ = image.shape

    face_landmarks_list = None
    hand_landmarks_list = None

    # ── DETECCIÓN CON MEDIAPIPE ─────────────────────────────
    if MEDIAPIPE_NUEVO:
        _ts_mp += 33  # Simula ~30 fps
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        rf = face_mesh.detect_for_video(mp_image, _ts_mp)
        if rf.face_landmarks:
            face_landmarks_list = rf.face_landmarks

        rh = hands.detect_for_video(mp_image, _ts_mp)
        if rh.hand_landmarks:
            hand_landmarks_list = rh.hand_landmarks
    else:
        rf = face_mesh.process(rgb_image)
        if rf.multi_face_landmarks:
            face_landmarks_list = [fl.landmark for fl in rf.multi_face_landmarks]
        rh = hands.process(rgb_image)
        if rh.multi_hand_landmarks:
            hand_landmarks_list = [hl.landmark for hl in rh.multi_hand_landmarks]

    gesto_detectado = None

    # ── PROCESAR FACE MESH ──────────────────────────────────
    if face_landmarks_list:
        for face_landmarks in face_landmarks_list:
            # Dibuja los 468 landmarks de la cara como puntos verdes
            for lm in face_landmarks:
                x, y = int(lm.x * w_img), int(lm.y * h_img)
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

            # Puntos clave para detectar gestos:
            boca     = face_landmarks[13]   # Landmark 13 = boca
            oreja_izq = face_landmarks[234] # Landmark 234 = oreja izquierda
            oreja_der = face_landmarks[454] # Landmark 454 = oreja derecha

            # Círculos visuales que marcan las zonas de detección
            bx, by = int(boca.x * w_img), int(boca.y * h_img)
            cv2.circle(image, (bx, by), 15, (0, 0, 255), 2)  # Rojo: boca
            cv2.circle(image, (bx, by), 3, (0, 0, 255), -1)
            ox1, oy1 = int(oreja_izq.x * w_img), int(oreja_izq.y * h_img)
            ox2, oy2 = int(oreja_der.x * w_img), int(oreja_der.y * h_img)
            cv2.circle(image, (ox1, oy1), 20, (255, 0, 0), 2)  # Azul: orejas
            cv2.circle(image, (ox2, oy2), 20, (255, 0, 0), 2)

            # ── DETECCIÓN DE GESTOS ─────────────────────────

            # Topo Gigio: dos manos, cada dedo índice cerca de una oreja
            if hand_landmarks_list:
                if len(hand_landmarks_list) == 2:
                    i0 = hand_landmarks_list[0][8]   # Landmark 8 = punta del índice (mano 1)
                    i1 = hand_landmarks_list[1][8]   # Landmark 8 = punta del índice (mano 2)
                    # Distancia normalizada < 0.3 = está tocando la oreja
                    if (dist(i0, oreja_izq) < 0.3 and dist(i1, oreja_der) < 0.3) or \
                       (dist(i0, oreja_der) < 0.3 and dist(i1, oreja_izq) < 0.3):
                        gesto_detectado = "topo_gigio"

                # Dybala: una mano con el índice cerca de la boca
                if not gesto_detectado:
                    for mano in hand_landmarks_list:
                        if dist(mano[8], boca) < 0.25:   # Distancia < 0.25 = cerca de la boca
                            gesto_detectado = "dybala"
                            break

    # ── DIBUJAR MANOS ───────────────────────────────────────
    if hand_landmarks_list:
        for mano in hand_landmarks_list:
            # Dibuja los 21 landmarks de cada mano como puntos amarillos
            for i, lm in enumerate(mano):
                x, y = int(lm.x * w_img), int(lm.y * h_img)
                cv2.circle(image, (x, y), 4, (255, 255, 0), -1)
                cv2.circle(image, (x, y), 4, (0, 0, 0), 1)
            # Dibuja las conexiones entre landmarks (el esqueleto de la mano)
            for a, b in CONEXIONES_MANO:
                p1 = mano[a]
                p2 = mano[b]
                x1, y1 = int(p1.x * w_img), int(p1.y * h_img)
                x2, y2 = int(p2.x * w_img), int(p2.y * h_img)
                cv2.line(image, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # ── MOSTRAR GESTO EN PANTALLA ──────────────────────────
    if gesto_detectado:
        info = GESTOS[gesto_detectado]

        # Superpone el emoji en la esquina superior izquierda
        emoji_img = EMOJIS[gesto_detectado]
        eh, ew = emoji_img.shape[:2]
        x0, y0 = 20, 20
        alpha = emoji_img[:, :, 3] / 255.0  # Canal alpha para transparencia
        for c in range(3):
            image[y0:y0+eh, x0:x0+ew, c] = (
                image[y0:y0+eh, x0:x0+ew, c] * (1 - alpha) + emoji_img[:, :, c] * alpha
            )

        # Superpone la foto del jugador al costado derecho si existe
        if overlay_foto_lado(image, info["foto"]):
            pass  # Foto mostrada
    else:
        if face_landmarks_list:
            cv2.putText(image, "Esperando gesto...", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Muestra el resultado en una ventana
    cv2.imshow("Gestos con Emojis", image)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC para salir
        break

# ── CIERRE ─────────────────────────────────────────────────
face_mesh.close()
if hasattr(hands, 'close'):
    hands.close()
cap.release()
cv2.destroyAllWindows()
