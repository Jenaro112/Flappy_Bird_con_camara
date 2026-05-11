import pygame
import sys
import cv2
import math
import os
import mediapipe as mp

# ===========================================================================
# DETECCIÓN AUTOMÁTICA DE VERSIÓN DE MEDIAPIPE
# - Versión antigua (< 0.10): usa mp.solutions.hands
# - Versión nueva  (>= 0.10): usa mediapipe.tasks + modelo .task descargable
# ===========================================================================
MEDIAPIPE_NUEVO = not hasattr(mp, "solutions")

if MEDIAPIPE_NUEVO:
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.vision import HandLandmarkerOptions, RunningMode
    import urllib.request

    MODEL_PATH = "hand_landmarker.task"
    MODEL_URL  = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    if not os.path.exists(MODEL_PATH):
        print("Descargando modelo MediaPipe (solo la primera vez)...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Modelo descargado OK.")
        except Exception as e:
            print(f"\nERROR al descargar el modelo: {e}")
            print("Descargalo manualmente desde:")
            print(MODEL_URL)
            print(f"Guardalo como '{MODEL_PATH}' en la carpeta del juego.")
            sys.exit(1)

    _opts = HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hands  = mp_vision.HandLandmarker.create_from_options(_opts)
    _ts_mp = 0

else:
    hands = mp.solutions.hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

# Estas constantes las necesitamos ANTES de procesar frames
ANCHO, ALTO = 1280, 720

def procesar_frame_mp(frame_rgb):
    global _ts_mp
    if MEDIAPIPE_NUEVO:
        _ts_mp += 33
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = hands.detect_for_video(mp_image, _ts_mp)
        if results.hand_landmarks:
            lm = results.hand_landmarks[0][8]
            return int(lm.x * ANCHO), int(lm.y * ALTO), results.hand_landmarks[0]
    else:
        res = hands.process(frame_rgb)
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark[8]
            return int(lm.x * ANCHO), int(lm.y * ALTO), res.multi_hand_landmarks[0].landmark
    return 0, 0, None

# ===========================================================================
# PYGAME
# ===========================================================================
pygame.init()
pantalla      = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("Penales en La Bombonera")
reloj         = pygame.time.Clock()
fuente_grande = pygame.font.SysFont("Arial", 72, bold=True)
fuente_media  = pygame.font.SysFont("Arial", 40, bold=True)
fuente_chica  = pygame.font.SysFont("Arial", 28)

BLANCO   = (255, 255, 255)
NEGRO    = (0,   0,   0)
ROJO     = (220, 30,  30)
AMARILLO = (255, 215, 0)
VERDE    = (34,  139, 34)

# ===========================================================================
# ASSETS CON FALLBACK
# ===========================================================================
def cargar_imagen(path, tamaño, alpha=False):
    try:
        img = pygame.image.load(path)
        img = img.convert_alpha() if alpha else img.convert()
        return pygame.transform.scale(img, tamaño), True
    except Exception:
        surf = pygame.Surface(tamaño, pygame.SRCALPHA if alpha else 0)
        surf.fill((45, 110, 45))
        return surf, False

img_fondo,  _          = cargar_imagen("assets/La_12.png",  (ANCHO, ALTO))
img_pelota, pelota_png = cargar_imagen("assets/pelota2.png", (60, 60), alpha=True)

def dibujar_pelota_fallback(sup, x, y, radio=30):
    pygame.draw.circle(sup, BLANCO, (int(x), int(y)), radio)
    pygame.draw.circle(sup, NEGRO,  (int(x), int(y)), radio, 3)
    for ang in range(0, 360, 60):
        r = math.radians(ang)
        pygame.draw.circle(sup, NEGRO,
                           (int(x + math.cos(r)*radio*0.55),
                            int(y + math.sin(r)*radio*0.55)), 6)

# ===========================================================================
# ARCO
# ===========================================================================
ARCO_X     = 291
ARCO_Y     = 298
ARCO_ANCHO = 698
ARCO_ALTO  = 269
PENAL_Y    = ALTO - 140

def pelota_en_arco(px, py):
    return pygame.Rect(ARCO_X, ARCO_Y, ARCO_ANCHO, ARCO_ALTO).collidepoint(int(px), int(py))

def dibujar_arco(sup):
    g = 5
    pygame.draw.line(sup, BLANCO, (ARCO_X, ARCO_Y), (ARCO_X, ARCO_Y + ARCO_ALTO), g)
    pygame.draw.line(sup, BLANCO, (ARCO_X + ARCO_ANCHO, ARCO_Y), (ARCO_X + ARCO_ANCHO, ARCO_Y + ARCO_ALTO), g)
    pygame.draw.line(sup, BLANCO, (ARCO_X, ARCO_Y), (ARCO_X + ARCO_ANCHO, ARCO_Y), g)

# ===========================================================================
# PORTERO
# ===========================================================================
class Portero:
    def __init__(self):
        self.x      = ANCHO // 2
        self.y      = ARCO_Y + ARCO_ALTO // 2
        self.vel    = 5
        self.dir    = 1
        self.activo = False

    def mover(self):
        if not self.activo:
            return
        self.x += self.vel * self.dir
        if self.x >= ARCO_X + ARCO_ANCHO - 30 or self.x <= ARCO_X + 30:
            self.dir *= -1

    def ataja(self, px, py):
        rect = pygame.Rect(int(self.x - 25), int(self.y - 45), 50, 90)
        return rect.inflate(20, 20).collidepoint(int(px), int(py))

    def dibujar(self, sup):
        rx, ry = int(self.x - 25), int(self.y - 45)
        pygame.draw.rect(sup, AMARILLO, (rx, ry, 50, 90), border_radius=8)
        pygame.draw.circle(sup, (255, 200, 150), (int(self.x), ry - 18), 18)
        txt = fuente_chica.render("1", True, NEGRO)
        sup.blit(txt, txt.get_rect(center=(int(self.x), int(self.y))))

# ===========================================================================
# ESTADO DEL JUEGO
# ===========================================================================
ESPERANDO = "esperando"
VOLANDO   = "volando"
RESULTADO = "resultado"

def reset_pelota():
    return float(ANCHO // 2), float(PENAL_Y), 0.0, 0.0

pelota_x, pelota_y, vel_x, vel_y = reset_pelota()
radio = 30

goles = atajadas = afuera = penal_actual = 0
PENALES_MAX     = 5
estado          = ESPERANDO
portero         = Portero()
mensaje         = ""
color_msj       = AMARILLO
timer_msj       = 0
juego_terminado = False
dedo_x_ant = dedo_y_ant = 0

# ===========================================================================
# CÁMARA
# ===========================================================================
camara = None
for idx in [0, 1, 2]:
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if cap.isOpened():
        for _ in range(5):
            cap.read()
        ok, frame_prueba = cap.read()
        if ok and frame_prueba is not None and frame_prueba.any():
            camara = cap
            print(f"Cámara encontrada en índice {idx}")
            break
        cap.release()

if camara is None:
    print("ERROR: No se encontró ninguna cámara.")
    pygame.quit()
    sys.exit()

# ===========================================================================
# FUNCIONES HUD
# ===========================================================================
def dibujar_hud(sup):
    partes = [
        (f"Goles: {goles}", AMARILLO),
        (" | ", (80, 80, 80)),
        (f"Atajadas: {atajadas}", ROJO),
        (" | ", (80, 80, 80)),
        (f"Afuera: {afuera}", (180, 180, 180)),
        (" | ", (80, 80, 80)),
        (f"Penal: {penal_actual}/{PENALES_MAX}", BLANCO),
    ]
    ancho = sum(fuente_chica.render(t, True, c).get_width() for t, c in partes)
    alto = fuente_chica.render("A", True, BLANCO).get_height()
    bx = ANCHO - ancho - 24
    by = 10
    bg = pygame.Surface((ancho + 24, alto + 12), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 160))
    sup.blit(bg, (bx, by))
    x = bx + 12
    for texto, color in partes:
        txt = fuente_chica.render(texto, True, color)
        sup.blit(txt, (x, by + 6))
        x += txt.get_width()

def dibujar_mensaje_central(sup, texto, color):
    sombra = fuente_grande.render(texto, True, NEGRO)
    txt    = fuente_grande.render(texto, True, color)
    cx, cy = ANCHO // 2, ALTO // 2
    sup.blit(sombra, sombra.get_rect(center=(cx+3, cy+3)))
    sup.blit(txt,    txt.get_rect(center=(cx, cy)))

def dibujar_pantalla_final(sup):
    over = pygame.Surface((ANCHO, ALTO), pygame.SRCALPHA)
    over.fill((0, 0, 0, 180))
    sup.blit(over, (0, 0))
    lineas = [
        ("FINAL!",                                  AMARILLO, fuente_grande),
        (f"Goles: {goles} / {PENALES_MAX}",         BLANCO,   fuente_media),
        (f"Atajadas: {atajadas}   Afuera: {afuera}", BLANCO,   fuente_media),
        ("Presiona R para jugar de nuevo",           BLANCO,   fuente_chica),
    ]
    y = ALTO // 2 - 130
    for texto, color, fuente in lineas:
        s = fuente.render(texto, True, color)
        sup.blit(s, s.get_rect(center=(ANCHO // 2, y)))
        y += s.get_height() + 20

def dibujar_ayuda(sup):
    if estado == ESPERANDO and not juego_terminado:
        txt = fuente_chica.render(
            "Acerca tu dedo indice a la pelota y movelo hacia el arco", True, BLANCO)
        fondo = pygame.Surface((txt.get_width()+20, txt.get_height()+8), pygame.SRCALPHA)
        fondo.fill((0, 0, 0, 120))
        sup.blit(fondo, (ANCHO//2 - txt.get_width()//2 - 10, ALTO - 50))
        sup.blit(txt, txt.get_rect(center=(ANCHO//2, ALTO - 42)))

# ===========================================================================
# BUCLE PRINCIPAL
# ===========================================================================
corriendo = True

while corriendo:
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            corriendo = False
        if evento.type == pygame.KEYDOWN:
            if evento.key == pygame.K_ESCAPE:
                corriendo = False
            if evento.key == pygame.K_r and juego_terminado:
                goles = atajadas = afuera = penal_actual = 0
                pelota_x, pelota_y, vel_x, vel_y = reset_pelota()
                estado = ESPERANDO
                portero = Portero()
                juego_terminado = False

    # ── CÁMARA ──────────────────────────────────────────────────────────────
    exito, frame = camara.read()
    dedo_x, dedo_y = 0, 0
    landmarks = None
    cam_surf = None

    if exito:
        frame     = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dedo_x, dedo_y, landmarks = procesar_frame_mp(frame_rgb)

        frame_mostrar = cv2.resize(frame_rgb, (320, 240))
        if landmarks:
            for lm in landmarks:
                cx, cy = int(lm.x * 320), int(lm.y * 240)
                cv2.circle(frame_mostrar, (cx, cy), 2, (0, 255, 0), -1)
            lm_tip = landmarks[8]
            cx, cy = int(lm_tip.x * 320), int(lm_tip.y * 240)
            cv2.circle(frame_mostrar, (cx, cy), 6, (255, 0, 0), -1)
        cam_surf = pygame.surfarray.make_surface(frame_mostrar.swapaxes(0, 1))

    # ── LÓGICA ──────────────────────────────────────────────────────────────
    if not juego_terminado:
        if estado == ESPERANDO and dedo_x != 0:
            dist = math.hypot(dedo_x - pelota_x, dedo_y - pelota_y)
            if dist <= 120:
                vx = (dedo_x - dedo_x_ant) * 0.9
                vy = (dedo_y - dedo_y_ant) * 0.9
                if math.hypot(vx, vy) > 5:
                    vel_x, vel_y   = vx, vy
                    estado         = VOLANDO
                    portero.activo = True
                    penal_actual  += 1

        elif estado == VOLANDO:
            pelota_x += vel_x
            pelota_y += vel_y
            vel_x *= 0.97
            vel_y *= 0.97
            portero.mover()

            resultado = None
            if portero.ataja(pelota_x, pelota_y):
                resultado = "atajada"
            elif pelota_y < ARCO_Y - 30 or pelota_x < 0 or pelota_x > ANCHO:
                resultado = "afuera"
            elif pelota_en_arco(pelota_x, pelota_y):
                resultado = "gol"

            if resultado:
                estado    = RESULTADO
                timer_msj = 120
                if resultado == "gol":
                    goles    += 1
                    mensaje   = "GOL!"
                    color_msj = AMARILLO
                elif resultado == "atajada":
                    atajadas += 1
                    mensaje   = "ATAJADA!"
                    color_msj = ROJO
                else:
                    afuera   += 1
                    mensaje   = "AFUERA!"
                    color_msj = BLANCO
                if penal_actual >= PENALES_MAX:
                    juego_terminado = True

        elif estado == RESULTADO:
            timer_msj -= 1
            if timer_msj <= 0:
                pelota_x, pelota_y, vel_x, vel_y = reset_pelota()
                estado         = ESPERANDO
                portero.activo = False

    if dedo_x != 0:
        dedo_x_ant, dedo_y_ant = dedo_x, dedo_y

    # ── RENDER ──────────────────────────────────────────────────────────────
    pantalla.blit(img_fondo, (0, 0))
    if cam_surf:
        pantalla.blit(cam_surf, (0, 0))
    dibujar_arco(pantalla)

    if not juego_terminado:
        portero.dibujar(pantalla)

    # Pelota con perspectiva
    if estado == VOLANDO:
        prog   = max(0.0, min(1.0, (PENAL_Y - pelota_y) / max(1, PENAL_Y - ARCO_Y)))
        escala = 1.0 - prog * 0.55
        tam    = max(10, int(60 * escala))
    else:
        tam = 60

    if pelota_png:
        ps = pygame.transform.scale(img_pelota, (tam, tam))
        pantalla.blit(ps, ps.get_rect(center=(int(pelota_x), int(pelota_y))))
    else:
        dibujar_pelota_fallback(pantalla, pelota_x, pelota_y, tam // 2)

    if dedo_x != 0 and dedo_y != 0:
        pygame.draw.circle(pantalla, ROJO,   (dedo_x, dedo_y), 12)
        pygame.draw.circle(pantalla, BLANCO, (dedo_x, dedo_y), 12, 2)

    dibujar_hud(pantalla)
    dibujar_ayuda(pantalla)

    if estado == RESULTADO and timer_msj > 0:
        dibujar_mensaje_central(pantalla, mensaje, color_msj)

    if juego_terminado:
        dibujar_pantalla_final(pantalla)

    pygame.display.flip()
    reloj.tick(60)

# ===========================================================================
# CIERRE
# ===========================================================================
camara.release()
hands.close()
pygame.quit()
sys.exit()