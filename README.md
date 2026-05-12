# ⚽ Penales & Flappy Bird por Voz

Proyecto interactivo con cámara para detectar gestos faciales y de manos usando **MediaPipe**. Incluye dos juegos y un detector de gestos.

## 📁 Archivos

### `game.py` — Flappy Bird controlado con la nariz
Controlás un pájaro moviendo la cabeza arriba/abajo. La cámara es el fondo del juego.

- **Detección**: Face Mesh de MediaPipe (landmark 1 = punta de la nariz)
- **Control**: la posición Y de la nariz mapea directamente a la altura del pájaro
- **Dificultad**: la velocidad aumenta cada 8 segundos, el espacio entre tuberías se reduce cada 10 segundos
- **Hard mode**: al llegar a 50 puntos, se vuelve extremadamente difícil

**Teclas:**
| Tecla | Acción |
|-------|--------|
| Cualquier tecla | Comenzar partida |
| `R` | Reiniciar tras game over |
| `ESC` | Salir |

### `prueba_camara.py` — Detector de gestos faciales + manos
Muestra la cámara con detección de rostro y manos en tiempo real. Reconoce dos gestos:

- **🐭 Topo Gigio (Riquelme)**: ambas manos cerca de las orejas → muestra foto de Riquelme
- **⭐ Dybala**: dedo índice cerca de la boca → muestra foto de Dybala

**Características:**
- Face Mesh: 468 puntos verdes dibujados sobre la cara
- Hand Landmarks: 21 puntos amarillos con conexiones por mano
- Círculos guía: boca (rojo), orejas (azul)
- Emoji del gesto detectado arriba a la izquierda
- Foto del jugador al costado (tamaño 550×550)

**Teclas:** `ESC` para salir

*Nota: requiere las imágenes `assets/riquelme.jpg` y `assets/dybala.jpeg`*

---

## 🚀 Cómo ejecutar

```bash
# 1. Activar entorno virtual (recomendado)
source venv/bin/activate

# 2. Jugar al Flappy Bird
python game.py

# 3. Probar detección de gestos
python prueba_camara.py
```

---

## 📦 Dependencias

### Instalación rápida

```bash
pip install opencv-python mediapipe pygame Pillow numpy
```

### Librerías requeridas

| Librería | Versión mínima | Uso |
|----------|---------------|-----|
| `opencv-python` | 4.5+ | Captura y procesamiento de cámara |
| `mediapipe` | 0.10+ | Face Mesh y Hand Landmarker |
| `pygame` | 2.0+ | Ventana de juego, renderizado, texturas |
| `Pillow` | 9.0+ | Renderizado de emojis con tipografía Apple |
| `numpy` | 1.21+ | Manipulación de arrays para imágenes |

*Pillow solo es necesaria para `prueba_camara.py` (renderizado de emojis).*  
*Numpy ya viene como dependencia de opencv-python.*

---

## 🧠 Cómo funciona internamente

### Face Mesh (468 landmarks)
MediaPipe detecta 468 puntos de referencia en la cara. Cada punto tiene coordenadas `(x, y, z)` normalizadas entre 0 y 1.

Puntos clave usados:
- **Landmark 1**: punta de la nariz (control del pájaro en game.py)
- **Landmark 13**: boca (detección de Dybala)
- **Landmark 234**: oreja izquierda
- **Landmark 454**: oreja derecha

### Hand Landmarks (21 landmarks por mano)
Cada mano tiene 21 puntos. El punto **8** es la punta del dedo índice. La detección de gestos compara distancias normalizadas entre puntos de la mano y puntos de la cara:

- **Topo Gigio**: los dedos índices de ambas manos están a menos de **0.3** de distancia de las orejas
- **Dybala**: un dedo índice está a menos de **0.25** de distancia de la boca

### Control del pájaro (game.py)
```python
target_y = nose_y_norm * ALTO        # posición Y de la nariz → coordenada de pantalla
pajarito_y += (target_y - pajarito_y) * 0.3  # suavizado
```

El pájaro sigue la nariz con un factor de suavizado de 0.3. Si no se detecta la cara, mantiene la última posición.

### Dificultad progresiva
| Tiempo | Velocidad tuberías | Gap (espacio) | Intervalo entre pipes |
|--------|-------------------|---------------|----------------------|
| 0-8s | 5 | 215px | 1600ms |
| 8-16s | 7 | 215px | 1600ms |
| 16-24s | 9 | 195px | 1500ms |
| 24-32s | 11 | 175px | 1400ms |
| ... | +2 c/8s | -20 c/10s | -100 c/10s |
| Score ≥ 50 | +4 extra | -30 extra (mín 120) | -200 extra (mín 700ms) |

### High score persistente
Se guarda en `high_score.json` automáticamente al perder o cerrar el juego.

---

## 🖼️ Assets necesarios

### Para `game.py` (en `assets/`)
| Archivo | Descripción |
|---------|-------------|
| `pipe-green.png` | Textura de tubería |
| `yellowbird-upflap.png` | Pájaro subiendo |
| `yellowbird-midflap.png` | Pájaro nivelado |
| `yellowbird-downflap.png` | Pájaro bajando |
| `base.png` | Suelo |
| `0.png` a `9.png` | Números para el puntaje |

### Para `prueba_camara.py` (en `assets/`)
| Archivo | Descripción |
|---------|-------------|
| `riquelme.jpg` | Foto de Riquelme |
| `dybala.jpeg` | Foto de Dybala |

### Modelos MediaPipe (se descargan solos al ejecutar)
| Archivo | Descripción |
|---------|-------------|
| `face_landmarker.task` | Modelo de Face Mesh |
| `hand_landmarker.task` | Modelo de Hand Landmarker |

---

## ⚙️ Notas técnicas

- **macOS**: si la cámara no se abre, asegurate de que la Terminal tenga permiso de cámara en Preferencias del Sistema → Privacidad → Cámara
- **Apple Silicon (M1/M2/M3/M4)**: MediaPipe corre con aceleración Metal automáticamente
- Los modelos `.task` se descargan una sola vez (~10 MB cada uno) de Google Storage
- Si alguna textura falta en `assets/`, el juego dibuja formas de colores como fallback
