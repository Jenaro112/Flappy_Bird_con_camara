# ============================================================
# FLAPPY BIRD - Controlado con la Nariz
# ============================================================
# El jugador controla un pájaro moviendo la cabeza arriba/abajo.
# La cámara en vivo se usa como fondo del juego.
# MediaPipe Face Mesh detecta la punta de la nariz (landmark 1)
# y la posición Y de la nariz controla la altura del pájaro.
# ============================================================

import pygame            # Ventanas, gráficos, teclado
import cv2               # Cámara y procesamiento de imágenes
import mediapipe as mp   # Detección facial (Face Mesh)
import sys               # Para salir del programa
import math              # No se usa directamente, pero disponible
import random            # Para generar posiciones aleatorias de tuberías
import os                # Para verificar si archivos existen
import json              # Para guardar/cargar el high score
import time              # Para medir tiempo de juego

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

# Colores (en formato RGB)
BLANCO = (255, 255, 255)
NEGRO  = (0, 0, 0)
CREMA  = (245, 230, 200)

# ── INICIALIZAR CÁMARA ──────────────────────────────────────
# Busca la cámara en los índices 0, 1, 2 (webcam integrada o externa)
camara = None
for idx in [0, 1, 2]:
    # En macOS usa AVFOUNDATION; si no está disponible, usa el backend default
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)
    if cap.isOpened():
        # Lee 5 frames de calentamiento (la cámara tarda en estabilizarse)
        for _ in range(5):
            cap.read()
        ok, frame_prueba = cap.read()
        # Verifica que el frame no sea negro (cámaras virtuales inválidas)
        if ok and frame_prueba is not None and frame_prueba.any():
            camara = cap
            print(f"Cámara en índice {idx}")
            break
        cap.release()

if camara is None:
    print("ERROR: No hay cámara.")
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
pipe_src = cargar_img("assets/pipe-green.png")
if pipe_src:
    # Escala la textura al doble para que sea más visible
    ESCALA_PIPE = 2.0
    pipe_src = pygame.transform.scale(pipe_src, (
        int(pipe_src.get_width() * ESCALA_PIPE),
        int(pipe_src.get_height() * ESCALA_PIPE)
    ))
    PIPE_ANCHO = pipe_src.get_width()
    PIPE_ALTO  = pipe_src.get_height()
    # El cap (cabezal más ancho) ocupa el ~12% del alto total de la imagen
    CAP_ALTO = int(PIPE_ALTO * 0.12)
    pipe_cap = pipe_src.subsurface(0, 0, PIPE_ANCHO, CAP_ALTO)          # Cabezal de la tubería
    pipe_body = pipe_src.subsurface(0, CAP_ALTO, PIPE_ANCHO, PIPE_ALTO - CAP_ALTO)  # Cuerpo
    pipe_cap_inv = pygame.transform.flip(pipe_cap, False, True)         # Cabezal invertido para tubería superior
    pipe_body_inv = pygame.transform.flip(pipe_body, False, True)       # Cuerpo invertido para tubería superior
else:
    PIPE_ANCHO, PIPE_ALTO = 90, 320
    CAP_ALTO = 30
    pipe_cap = pipe_body = None  # Modo fallback: dibuja rectángulos

# --- Pájaro (3 texturas para animación de aleteo) ---
bird_up   = cargar_img("assets/yellowbird-upflap.png")   # Subiendo
bird_mid  = cargar_img("assets/yellowbird-midflap.png")   # Nivelado
bird_down = cargar_img("assets/yellowbird-downflap.png")  # Bajando
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
base_img = cargar_img("assets/base.png")
if base_img:
    BASE_ALTO = base_img.get_height()  # Altura del suelo en píxeles
else:
    BASE_ALTO = 80

# --- Números del puntaje (0.png a 9.png) ---
nums = []
nums_ok = True
for i in range(10):
    n = cargar_img(f"assets/{i}.png")
    if n is None:
        nums_ok = False
        break
    nums.append(n)
if not nums_ok:
    nums = []  # Fallback: usa texto de pygame

# ── CONSTANTES DE JUEGO ────────────────────────────────────
GAP_ALTO          = 280    # Espacio vertical entre tubería superior e inferior
GAP_ALTO_MIN      = 155    # Gap mínimo (cuando la dificultad sube)
PIPE_VEL_INICIAL  = 5      # Velocidad horizontal inicial de las tuberías
PIPE_INTERVALO_MS = 2000   # Milisegundos entre cada par de tuberías
PIPE_INTERVALO_MIN = 1000  # Intervalo mínimo
BIRD_X            = 250    # Posición X fija del pájaro (no se mueve horizontalmente)
SUAVIDAD          = 0.3    # Factor de suavizado: 0 = lento, 1 = instantáneo
SUELO_Y           = ALTO - BASE_ALTO  # Límite inferior del área de juego

# ── ESTADO DEL JUEGO ─────────────────────────────────────────
# High score: se guarda en un archivo JSON para que persista entre partidas
def cargar_high_score():
    try:
        with open("high_score.json") as f:
            return json.load(f).get("high_score", 0)
    except:
        return 0

def guardar_high_score(valor):
    with open("high_score.json", "w") as f:
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
    pipe = {"x": ANCHO, "top_h": th, "gap": gap, "paso": False}

    # Pre-renderiza las superficies de las tuberías (una sola vez, no por frame)
    if pipe_cap:
        gap_y = th + gap
        bot_h = SUELO_Y - gap_y

        # Tubería superior: cabezal abajo, cuerpo hacia arriba
        top = pygame.Surface((PIPE_ANCHO, th), pygame.SRCALPHA)
        top.blit(pipe_cap_inv, (0, th - CAP_ALTO))
        if th > CAP_ALTO:
            top.blit(pygame.transform.scale(pipe_body_inv, (PIPE_ANCHO, th - CAP_ALTO)), (0, 0))
        pipe["top_img"] = top

        # Tubería inferior: cabezal arriba, cuerpo hacia abajo
        bot = pygame.Surface((PIPE_ANCHO, bot_h), pygame.SRCALPHA)
        bot.blit(pipe_cap, (0, 0))
        if bot_h > CAP_ALTO:
            bot.blit(pygame.transform.scale(pipe_body, (PIPE_ANCHO, bot_h - CAP_ALTO)), (0, CAP_ALTO))
        pipe["bot_img"] = bot

    return pipe

def dibujar_pipe(sup, pipe):
    """Dibuja una tubería en la pantalla. Usa las superficies pre-renderizadas si existen."""
    if "top_img" in pipe:
        sup.blit(pipe["top_img"], (pipe["x"], 0))
        sup.blit(pipe["bot_img"], (pipe["x"], pipe["top_h"] + pipe["gap"]))
    else:
        # Fallback: dibuja rectángulos verdes si no hay textura
        th = pipe["top_h"]
        gap_y = th + pipe["gap"]
        pygame.draw.rect(sup, (34, 139, 34), (pipe["x"], 0, PIPE_ANCHO, th))
        pygame.draw.rect(sup, (20, 100, 20), (pipe["x"] - 4, th - 30, PIPE_ANCHO + 8, 30))
        pygame.draw.rect(sup, (34, 139, 34), (pipe["x"], gap_y, PIPE_ANCHO, SUELO_Y - gap_y))
        pygame.draw.rect(sup, (20, 100, 20), (pipe["x"] - 4, gap_y, PIPE_ANCHO + 8, 30))

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
    fuente_s = pygame.font.SysFont("impact", 58)  # Para el puntaje grande
    fuente_m = pygame.font.SysFont("impact", 32)  # Para SCORE y BEST
    fuente_p = pygame.font.SysFont("comicsansms", 22)  # Para los segundos

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

# ── BUCLE PRINCIPAL ──────────────────────────────────────────
while True:
    dt = reloj.tick(60)  # Limita a 60 FPS
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
        pajarito_y = pajarito_y + diff * SUAVIDAD

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
        vel_pipes = PIPE_VEL_INICIAL + max(0, segundos - 15) // 12 * 2
        # Gap: se reduce cada 10 segundos después de los primeros 20s
        gap_actual = max(GAP_ALTO_MIN, GAP_ALTO - max(0, segundos - 20) // 10 * 20)
        # Hard mode: al llegar a 50 puntos, dificultad extra
        if score >= 50:
            vel_pipes += 4
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
            pipe["x"] -= vel_pipes  # Mover hacia la izquierda

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
        pantalla.blit(pygame.transform.scale(base_img, (ANCHO, BASE_ALTO)), (0, SUELO_Y))
    else:
        pygame.draw.rect(pantalla, (100, 180, 50), (0, SUELO_Y, ANCHO, BASE_ALTO))

    # Menú principal
    if estado == "menu":
        pantalla.fill(CREMA)
        fuente_menu = pygame.font.SysFont("impact", 90)
        titulo = fuente_menu.render("FLAPPY BIRD", True, (60, 35, 10))
        sombra_t = fuente_menu.render("FLAPPY BIRD", True, NEGRO)
        pantalla.blit(sombra_t, sombra_t.get_rect(center=(ANCHO // 2 + 4, ALTO // 2 - 96)))
        pantalla.blit(titulo, titulo.get_rect(center=(ANCHO // 2, ALTO // 2 - 100)))
        if bird_ok:
            pantalla.blit(bird_mid, bird_mid.get_rect(center=(ANCHO // 2, ALTO // 2 + 10)))
        fuente_sub = pygame.font.SysFont("comicsansms", 26)
        subtitulo = fuente_sub.render("Presiona cualquier tecla para comenzar", True, (120, 90, 50))
        pantalla.blit(subtitulo, subtitulo.get_rect(center=(ANCHO // 2, ALTO // 2 + 90)))
        control = fuente_sub.render("Movete arriba/abajo para controlar el pajaro", True, (160, 130, 90))
        pantalla.blit(control, control.get_rect(center=(ANCHO // 2, ALTO // 2 + 130)))

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
