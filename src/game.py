# ============================================================
# FLAPPY BIRD - Controlado con la Nariz
# ============================================================
# El jugador controla un pájaro moviendo la cabeza arriba/abajo.
# La cámara en vivo se usa como fondo del juego.
# MediaPipe Face Mesh detecta la punta de la nariz (landmark 1)
# y la posición Y de la nariz controla la altura del pájaro.
# ============================================================

import os
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import sys
_stderr = sys.stderr
sys.stderr = open(os.devnull, "w")

import pygame            # Ventanas, gráficos, teclado
import cv2               # Cámara y procesamiento de imágenes
import mediapipe as mp   # Detección facial (Face Mesh)
import math              # No se usa directamente, pero disponible
import random            # Para generar posiciones aleatorias de tuberías
import json              # Para guardar/cargar el high score
import time              # Para medir tiempo de juego

sys.stderr = _stderr

# ── RESOLUCIÓN DE RUTAS ─────────────────────────────────────
# Permite ejecutar el script desde cualquier directorio.
# _PROJECT_ROOT apunta siempre a la raíz del proyecto (un nivel arriba de src/).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _ruta(*partes):
    """Construye una ruta absoluta relativa a la raíz del proyecto."""
    return os.path.join(_PROJECT_ROOT, *partes)

# ── DETECCIÓN AUTOMÁTICA DE VERSIÓN DE MEDIAPIPE ────────────
# MediaPipe versión nueva (>= 0.10) usa mediapipe.tasks con archivos .task
# MediaPipe versión antigua (< 0.10) usa mp.solutions
MEDIAPIPE_NUEVO = not hasattr(mp, "solutions")

if MEDIAPIPE_NUEVO:
    import urllib.request
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.vision import FaceLandmarkerOptions, RunningMode

    # Si no existe el modelo .task, lo descarga de Google Storage
    MODEL_PATH = _ruta("models", "face_landmarker.task")
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    if not os.path.exists(MODEL_PATH):
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception:
            sys.exit(1)

    # Configuración del Face Landmarker en modo VIDEO (procesa frame por frame)
    face_opts = FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_faces=1
    )
    face_mesh = mp_vision.FaceLandmarker.create_from_options(face_opts)
    _ts_mp = 0  # Timestamp interno para MediaPipe
else:
    # Versión antigua de MediaPipe (< 0.10)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# ── INICIALIZAR PYGAME ──────────────────────────────────────
pygame.init()
ANCHO, ALTO = 1280, 720                     # Resolución de la ventana
pantalla = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("Flappy Bird - Control con la Nariz")
reloj = pygame.time.Clock()                  # Controla los FPS
fuente_info = pygame.font.SysFont("Arial", 28)
# Fuentes cacheadas para el HUD (evita recrearlas cada frame)
fuente_score  = pygame.font.SysFont("impact", 58)
fuente_best   = pygame.font.SysFont("impact", 32)
fuente_tiempo = pygame.font.SysFont("comicsansms", 22)

# Colores (en formato RGB)
BLANCO = (255, 255, 255)
NEGRO  = (0, 0, 0)
CREMA  = (245, 230, 200)

# ── INICIALIZAR CÁMARA ──────────────────────────────────────
# Prioridad: índices 1, 2 primero (la webcam integrada del Mac suele
# desplazarse a estos cuando Continuity Camera del iPhone ocupa el 0).
# Luego 0 como fallback, y finalmente 3 por si hay cámaras externas.
camara = None
backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else 0

for idx in [1, 2, 0, 3]:
    cap = cv2.VideoCapture(idx, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)
    if cap.isOpened():
        # Lee 5 frames de calentamiento (la cámara tarda en estabilizarse)
        for _ in range(5):
            cap.read()
        ok, frame_prueba = cap.read()
        # Verifica que el frame no sea negro (cámaras virtuales inválidas)
        if ok and frame_prueba is not None and frame_prueba.any():
            # Validar dimensiones reales del frame
            h_test, w_test = frame_prueba.shape[:2]
            if w_test > 0 and h_test > 0:
                camara = cap
                print(f"Cámara seleccionada: índice {idx} ({w_test}x{h_test})")
                break
        cap.release()

if camara is None:
    print("ERROR: No se encontró ninguna cámara disponible.")
    sys.exit(1)

# ── CARGAR TEXTURAS ─────────────────────────────────────────
# Función auxiliar: carga una imagen PNG/JPG con transparencia y escala opcional
def cargar_img(ruta, escala=None):
    try:
        img = pygame.image.load(ruta).convert_alpha()
        if escala:
            img = pygame.transform.scale(img, escala)
        return img
    except:
        return None  # Si no existe el archivo, devuelve None

# --- Tubería (pipe-green.png) ---
pipe_src = cargar_img(_ruta("assets", "pipe-green.png"))
if pipe_src:
    # Escala la textura al doble para que sea más visible
    ESCALA_PIPE = 2.0
    pipe_src = pygame.transform.scale(pipe_src, (
        int(pipe_src.get_width() * ESCALA_PIPE),
        int(pipe_src.get_height() * ESCALA_PIPE)
    ))
    PIPE_ANCHO = pipe_src.get_width()
    PIPE_ALTO  = pipe_src.get_height()
    # El cap real en pipe-green.png ocupa los primeros ~26px (borde + detalle + sombra)
    CAP_PX_ORIGINAL = 26
    CAP_ALTO = int(CAP_PX_ORIGINAL * ESCALA_PIPE)
    pipe_body_h = PIPE_ALTO - CAP_ALTO
    # Piezas para tubería inferior (normales)
    pipe_cap = pipe_src.subsurface(0, 0, PIPE_ANCHO, CAP_ALTO)
    pipe_body = pipe_src.subsurface(0, CAP_ALTO, PIPE_ANCHO, pipe_body_h)
    # Tubería superior: invertir la imagen COMPLETA y luego tomar subsurfaces
    # (evita pygame.transform.flip sobre subsurfaces, que da problemas visuales)
    # En la imagen invertida, el cuerpo queda arriba (y=0) y el cap abajo (y=pipe_body_h)
    pipe_src_inv = pygame.transform.flip(pipe_src, False, True)
    pipe_body_inv = pipe_src_inv.subsurface(0, 0, PIPE_ANCHO, pipe_body_h)
    pipe_cap_inv = pipe_src_inv.subsurface(0, pipe_body_h, PIPE_ANCHO, CAP_ALTO)
else:
    PIPE_ANCHO, PIPE_ALTO = 90, 320
    CAP_ALTO = 30
    pipe_cap = pipe_body = None  # Modo fallback: dibuja rectángulos

# --- Pájaro (3 texturas para animación de aleteo) ---
bird_up   = cargar_img(_ruta("assets", "yellowbird-upflap.png"))   # Subiendo
bird_mid  = cargar_img(_ruta("assets", "yellowbird-midflap.png"))   # Nivelado
bird_down = cargar_img(_ruta("assets", "yellowbird-downflap.png"))  # Bajando
bird_ok   = all(x is not None for x in [bird_up, bird_mid, bird_down])
if bird_ok:
    ESCALA_PAJARO = 2.2  # Agranda el pájaro para que se vea bien
    bird_up   = pygame.transform.scale(bird_up,   (int(bird_up.get_width() * ESCALA_PAJARO), int(bird_up.get_height() * ESCALA_PAJARO)))
    bird_mid  = pygame.transform.scale(bird_mid,  (int(bird_mid.get_width() * ESCALA_PAJARO), int(bird_mid.get_height() * ESCALA_PAJARO)))
    bird_down = pygame.transform.scale(bird_down, (int(bird_down.get_width() * ESCALA_PAJARO), int(bird_down.get_height() * ESCALA_PAJARO)))
    BIRD_W = bird_up.get_width()
    BIRD_H = bird_up.get_height()
    BIRD_RADIO = max(BIRD_W, BIRD_H) // 2  # Radio usado para colisiones
else:
    BIRD_W = BIRD_H = 36
    BIRD_RADIO = 18

# --- Base / suelo (base.png) ---
base_img = cargar_img(_ruta("assets", "base.png"))
if base_img:
    BASE_ALTO = base_img.get_height()  # Altura del suelo en píxeles
    base_img = pygame.transform.scale(base_img, (ANCHO, BASE_ALTO))  # Pre-escalar una vez
else:
    BASE_ALTO = 80

# --- Números del puntaje (0.png a 9.png) ---
nums = []
nums_ok = True
for i in range(10):
    n = cargar_img(_ruta("assets", f"{i}.png"))
    if n is None:
        nums_ok = False
        break
    nums.append(n)
if not nums_ok:
    nums = []  # Fallback: usa texto de pygame

# --- Menú (Menu.png) ---
menu_img = cargar_img(_ruta("assets", "Menu.png"))
if menu_img:
    menu_img = pygame.transform.scale(menu_img, (ANCHO, ALTO))  # Pre-escalar una vez

# ── CONSTANTES DE JUEGO ────────────────────────────────────
GAP_ALTO          = 280    # Espacio vertical entre tubería superior e inferior
GAP_ALTO_MIN      = 155    # Gap mínimo (cuando la dificultad sube)
PIPE_VEL_INICIAL  = 300    # Velocidad horizontal en píxeles/SEGUNDO (5 px/frame × 60 fps)
PIPE_INTERVALO_MS = 2000   # Milisegundos entre cada par de tuberías
PIPE_INTERVALO_MIN = 1000  # Intervalo mínimo
BIRD_X            = 250    # Posición X fija del pájaro (no se mueve horizontalmente)
SUAVIDAD          = 0.3    # Factor de suavizado: 0 = lento, 1 = instantáneo
SUELO_Y           = ALTO - BASE_ALTO  # Límite inferior del área de juego

# ── ESTADO DEL JUEGO ─────────────────────────────────────────
# High score: se guarda en un archivo JSON para que persista entre partidas
def cargar_high_score():
    try:
        with open(_ruta("data", "high_score.json")) as f:
            return json.load(f).get("high_score", 0)
    except:
        return 0

def guardar_high_score(valor):
    with open(_ruta("data", "high_score.json"), "w") as f:
        json.dump({"high_score": valor}, f)

pajarito_y    = ALTO // 2     # Posición Y inicial del pájaro
pajarito_flap = "mid"         # Estado del aleteo: "up", "mid", "down"
score         = 0
high_score    = cargar_high_score()  # Récord guardado
tiempo_inicio = 0             # Tiempo de inicio de la partida actual
vel_pipes     = PIPE_VEL_INICIAL
gap_actual    = GAP_ALTO      # Gap actual (se reduce con el tiempo)
pipes         = []            # Lista de tuberías activas
ultimo_pipe_ms = 0            # Último momento en que se generó una tubería
estado        = "menu"        # "menu" | "jugando" | "game_over"
nose_y_norm   = 0.5           # Posición Y normalizada de la nariz (0 = arriba, 1 = abajo)
cam_surf      = None          # Superficie de pygame para el fondo de cámara

# ── FUNCIONES DEL JUEGO ──────────────────────────────────────

def generar_pipe():
    """Crea un nuevo par de tuberías (superior e inferior) en el borde derecho."""
    min_h = 80
    max_h = SUELO_Y - gap_actual - 80  # Deja espacio para el gap y el suelo
    th = random.randint(min_h, max_h)  # Altura de la tubería superior
    gap = gap_actual
    pipe = {"x": float(ANCHO), "top_h": th, "gap": gap, "paso": False}

    # Pre-renderiza las superficies de las tuberías (una sola vez, no por frame)
    if pipe_cap:
        gap_y = th + gap
        bot_h = SUELO_Y - gap_y

        # Tubería superior: cabezal abajo, cuerpo hacia arriba
        top = pygame.Surface((PIPE_ANCHO, th)).convert_alpha()
        top.fill((0, 0, 0, 0))
        top.blit(pipe_cap_inv, (0, th - CAP_ALTO))
        if th > CAP_ALTO:
            top.blit(pygame.transform.scale(pipe_body_inv, (PIPE_ANCHO, th - CAP_ALTO)), (0, 0))
        pipe["top_img"] = top

        # Tubería inferior: cabezal arriba, cuerpo hacia abajo
        bot = pygame.Surface((PIPE_ANCHO, bot_h)).convert_alpha()
        bot.fill((0, 0, 0, 0))
        bot.blit(pipe_cap, (0, 0))
        if bot_h > CAP_ALTO:
            bot.blit(pygame.transform.scale(pipe_body, (PIPE_ANCHO, bot_h - CAP_ALTO)), (0, CAP_ALTO))
        pipe["bot_img"] = bot

    return pipe

def dibujar_pipe(sup, pipe):
    """Dibuja una tubería. Convierte x a int solo para el blit (evita sub-pixel jitter)."""
    px = int(pipe["x"])  # Float → int únicamente para renderizado
    if "top_img" in pipe:
        sup.blit(pipe["top_img"], (px, 0))
        sup.blit(pipe["bot_img"], (px, pipe["top_h"] + pipe["gap"]))
    else:
        # Fallback: dibuja rectángulos verdes si no hay textura
        th = pipe["top_h"]
        gap_y = th + pipe["gap"]
        pygame.draw.rect(sup, (34, 139, 34), (px, 0, PIPE_ANCHO, th))
        pygame.draw.rect(sup, (20, 100, 20), (px - 4, th - 30, PIPE_ANCHO + 8, 30))
        pygame.draw.rect(sup, (34, 139, 34), (px, gap_y, PIPE_ANCHO, SUELO_Y - gap_y))
        pygame.draw.rect(sup, (20, 100, 20), (px - 4, gap_y, PIPE_ANCHO + 8, 30))

def dibujar_pajarito(sup, x, y, flap):
    """Dibuja el pájaro en la posición (x, y) con la textura según el aleteo."""
    if bird_ok:
        imgs = {"up": bird_up, "mid": bird_mid, "down": bird_down}
        img = imgs.get(flap, bird_mid)
        sup.blit(img, img.get_rect(center=(x, y)))
    else:
        # Fallback: dibuja un círculo amarillo con pico
        dx, dy = 6, -4
        pygame.draw.circle(sup, (255, 215, 0), (x, y), BIRD_RADIO)
        pygame.draw.circle(sup, NEGRO, (x, y), BIRD_RADIO, 2)
        pygame.draw.circle(sup, NEGRO, (x + dx, y + dy), 4)
        pygame.draw.circle(sup, BLANCO, (x + dx, y + dy), 2)
        pygame.draw.polygon(sup, (255, 140, 0), [
            (x + BIRD_RADIO, y), (x + BIRD_RADIO + 14, y - 2), (x + BIRD_RADIO + 14, y + 6)
        ])

def dibujar_hud(sup):
    """Dibuja la interfaz: puntuación (centrada) y panel izquierdo (score, best, tiempo)."""
    segundos = max(0, int(time.time() - tiempo_inicio))
    fuente_s = fuente_score   # Puntaje grande (cacheada al inicio)
    fuente_m = fuente_best    # SCORE y BEST (cacheada al inicio)
    fuente_p = fuente_tiempo  # Segundos (cacheada al inicio)

    # Puntaje grande centrado arriba (usa imágenes 0.png-9.png si existen)
    if nums:
        digitos = [int(d) for d in str(score)]
        ancho_total = sum(nums[d].get_width() for d in digitos)
        x = (ANCHO - ancho_total) // 2
        for d in digitos:
            sup.blit(nums[d], (x, 14))
            x += nums[d].get_width()
    else:
        txt = fuente_s.render(str(score), True, BLANCO)
        sombra = fuente_s.render(str(score), True, NEGRO)
        sup.blit(sombra, (ANCHO // 2 - sombra.get_width() // 2 + 3, 17))
        sup.blit(txt, (ANCHO // 2 - txt.get_width() // 2, 14))

    # Panel izquierdo con SCORE, BEST, y tiempo
    bg = pygame.Surface((240, 100), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 130))
    sup.blit(bg, (12, 12))

    r1 = fuente_m.render(f"SCORE: {score}", True, BLANCO)
    r2 = fuente_m.render(f"BEST: {high_score}", True, (255, 215, 0))
    r3 = fuente_p.render(f"{segundos}s", True, (200, 200, 200))
    sup.blit(r1, (20, 14))
    sup.blit(r2, (20, 44))
    sup.blit(r3, (20, 72))

def reiniciar():
    """Reinicia todas las variables del juego para empezar una nueva partida."""
    global pajarito_y, pajarito_flap, score, tiempo_inicio, vel_pipes
    global pipes, ultimo_pipe_ms, estado, gap_actual, high_score
    if score > high_score:
        high_score = score
        guardar_high_score(high_score)
    pajarito_y = ALTO // 2
    pajarito_flap = "mid"
    score = 0
    tiempo_inicio = time.time()
    vel_pipes = PIPE_VEL_INICIAL
    gap_actual = GAP_ALTO
    pipes.clear()
    pipes.append(generar_pipe())  # Genera la primera tubería inmediatamente
    ultimo_pipe_ms = pygame.time.get_ticks()
    estado = "jugando"

print("El juego está corriendo correctamente")

# ── BUCLE PRINCIPAL ──────────────────────────────────────────
while True:
    dt_ms = reloj.tick(60)        # Limita a 60 FPS, devuelve ms reales del frame
    dt = dt_ms / 1000.0           # Convertir a segundos para cálculos de física
    ahora_ms = pygame.time.get_ticks()

    # ─── EVENTOS ────────────────────────────────────────────
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            camara.release()
            if hasattr(face_mesh, 'close'): face_mesh.close()
            pygame.quit()
            sys.exit()
        if evento.type == pygame.KEYDOWN:
            if evento.key == pygame.K_ESCAPE:
                camara.release()
                if hasattr(face_mesh, 'close'): face_mesh.close()
                pygame.quit()
                sys.exit()
            if estado == "menu":
                reiniciar()  # Cualquier tecla inicia la partida
            if evento.key == pygame.K_r and estado == "game_over":
                reiniciar()

    # ─── CÁMARA + DETECCIÓN FACIAL ─────────────────────────
    exito, frame = camara.read()
    if exito:
        frame = cv2.flip(frame, 1)                     # Espejo (efecto selfie)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV BGR → RGB

        # Face Mesh: detecta los 468 landmarks de la cara
        if MEDIAPIPE_NUEVO:
            _ts_mp += 33
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            rf = face_mesh.detect_for_video(mp_image, _ts_mp)
            if rf.face_landmarks:
                nose_y_norm = rf.face_landmarks[0][1].y  # Landmark 1 = punta de la nariz
        else:
            rf = face_mesh.process(frame_rgb)
            if rf.multi_face_landmarks:
                nose_y_norm = rf.multi_face_landmarks[0].landmark[1].y

        # Redimensiona el frame de cámara para usarlo como fondo del juego
        frame_fondo = cv2.resize(frame_rgb, (ANCHO, ALTO))
        # Convierte OpenCV (H,W,C) → Pygame (W,H,C) y crea una superficie
        cam_surf = pygame.surfarray.make_surface(frame_fondo.swapaxes(0, 1))

    # ─── LÓGICA DEL JUEGO ──────────────────────────────────
    if estado == "jugando":
        # La posición Y de la nariz controla la altura del pájaro con suavizado
        target_y = nose_y_norm * ALTO
        diff = target_y - pajarito_y
        # Suavizado exponencial independiente del framerate:
        # factor = 1 - (1 - SUAVIDAD)^(dt * 60) normaliza a 60 FPS de referencia
        factor = 1.0 - (1.0 - SUAVIDAD) ** (dt * 60.0)
        pajarito_y = pajarito_y + diff * factor

        # Elige la textura del pájaro según la dirección del movimiento
        if diff < -3:
            pajarito_flap = "up"
        elif diff > 3:
            pajarito_flap = "down"
        else:
            pajarito_flap = "mid"

        # ── DIFICULTAD PROGRESIVA ──────────────────────────
        # Velocidad: aumenta cada 12 segundos después de los primeros 15s
        segundos = max(0, int(time.time() - tiempo_inicio))
        vel_pipes = PIPE_VEL_INICIAL + max(0, segundos - 15) // 12 * 120  # +120 px/s c/12s
        # Gap: se reduce cada 10 segundos después de los primeros 20s
        gap_actual = max(GAP_ALTO_MIN, GAP_ALTO - max(0, segundos - 20) // 10 * 20)
        # Hard mode: al llegar a 50 puntos, dificultad extra
        if score >= 50:
            vel_pipes += 240  # +240 px/s (equivale a +4 px/frame a 60fps)
            gap_actual = max(120, gap_actual - 30)

        # Intervalo entre tuberías: se acorta cada 10s después de los primeros 20s
        intervalo = max(PIPE_INTERVALO_MIN, PIPE_INTERVALO_MS - max(0, segundos - 20) // 10 * 100)
        if score >= 50:
            intervalo = max(700, intervalo - 200)

        # Generar nueva tubería si pasó el intervalo
        if ahora_ms - ultimo_pipe_ms > intervalo:
            pipes.append(generar_pipe())
            ultimo_pipe_ms = ahora_ms

        # Mover tuberías y detectar colisiones
        nuevas = []
        for pipe in pipes:
            pipe["x"] -= vel_pipes * dt  # Movimiento independiente del framerate

            # Si la tubería salió de la pantalla, la descartamos
            if pipe["x"] + PIPE_ANCHO < 0:
                continue

            # Si el pájaro pasó la tubería, sumamos un punto
            if not pipe["paso"] and pipe["x"] + PIPE_ANCHO < BIRD_X:
                pipe["paso"] = True
                score += 1

            # Colisión: el pájaro choca contra la tubería
            if (BIRD_X + BIRD_RADIO > pipe["x"] and
                BIRD_X - BIRD_RADIO < pipe["x"] + PIPE_ANCHO):
                if (pajarito_y - BIRD_RADIO < pipe["top_h"] or
                    pajarito_y + BIRD_RADIO > pipe["top_h"] + pipe["gap"]):
                    estado = "game_over"
                    if score > high_score:
                        high_score = score
                        guardar_high_score(high_score)

            nuevas.append(pipe)

        # Colisión con bordes: techo o suelo
        if pajarito_y - BIRD_RADIO < 0 or pajarito_y + BIRD_RADIO > SUELO_Y:
            estado = "game_over"
            if score > high_score:
                high_score = score
                guardar_high_score(high_score)

        pipes = nuevas

    # ─── RENDERIZADO ────────────────────────────────────────
    # Fondo: la cámara en vivo
    if cam_surf:
        pantalla.blit(cam_surf, (0, 0))
    else:
        pantalla.fill((50, 50, 80))

    # Suelo (se dibuja abajo de todo)
    if base_img:
        pantalla.blit(base_img, (0, SUELO_Y))
    else:
        pygame.draw.rect(pantalla, (100, 180, 50), (0, SUELO_Y, ANCHO, BASE_ALTO))

    # Menú principal
    if estado == "menu":
        if menu_img:
            pantalla.blit(menu_img, (0, 0))
        else:
            pantalla.fill(CREMA)

    # Durante la partida o game over
    elif estado in ("jugando", "game_over"):
        for pipe in pipes:
            dibujar_pipe(pantalla, pipe)
        dibujar_pajarito(pantalla, BIRD_X, int(pajarito_y), pajarito_flap)
        dibujar_hud(pantalla)

    # Pantalla de game over
    if estado == "game_over":
        overlay = pygame.Surface((ANCHO, ALTO), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        pantalla.blit(overlay, (0, 0))
        txt = pygame.font.SysFont("impact", 60).render("GAME OVER", True, (255, 50, 50))
        pantalla.blit(txt, txt.get_rect(center=(ANCHO // 2, ALTO // 2 - 40)))
        txt2 = pygame.font.SysFont("comicsansms", 26).render("Presiona R para reiniciar", True, BLANCO)
        pantalla.blit(txt2, txt2.get_rect(center=(ANCHO // 2, ALTO // 2 + 30)))

    pygame.display.flip()  # Actualiza la ventana
