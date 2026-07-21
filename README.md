# 🐦 Flappy Bird — Control con Cámara

> Un clon de Flappy Bird donde controlás el pájaro moviendo tu cabeza. La cámara web detecta la punta de tu nariz en tiempo real y mapea su posición a la altura del pájaro. Incluye además un detector de gestos de celebración.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-2.6-green?logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange?logo=google)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Descripción

Este proyecto combina **visión por computadora** con un **videojuego clásico**:

- **Flappy Bird** (`src/game.py`): Controlás un pájaro con el movimiento de tu nariz. La cámara en vivo se usa como fondo del juego. MediaPipe Face Mesh detecta 468 puntos de la cara y usa el landmark 1 (punta de la nariz) para controlar la altura.

- **Detector de Gestos** (`src/gesture_detector.py`): Reconoce dos festejos de fútbol argentino en tiempo real:
  - 🐭 **Topo Gigio** (Riquelme): ambas manos cerca de las orejas
  - ⭐ **Dybala**: dedo índice en la boca

---

## ⚙️ Requisitos Previos

| Requisito | Versión mínima |
|-----------|---------------|
| Python | 3.9+ |
| Cámara web | Integrada o USB |
| Sistema operativo | macOS / Windows / Linux |
| Espacio en disco | ~50 MB (incluye modelos ML) |

> **macOS**: Asegurate de que la Terminal tenga permiso de cámara en *Preferencias del Sistema → Privacidad → Cámara*.

---

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/Jenaro112/Penales.git
cd Penales
```

### 2. Crear entorno virtual

```bash
# Crear el entorno virtual
python3 -m venv venv

# Activarlo
# macOS / Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

> Las versiones están congeladas en `requirements.txt` para garantizar compatibilidad. Si agregás nuevas librerías, regenerá el archivo con:
> ```bash
> pip freeze > requirements.txt
> ```

---

## 🎮 Ejecución

### Jugar al Flappy Bird

```bash
python main.py
```

O directamente:

```bash
python src/game.py
```

### Ejecutar el Detector de Gestos

```bash
python main.py gestures
```

O directamente:

```bash
python src/gesture_detector.py
```

> **Nota**: Los modelos de MediaPipe (~11 MB) se descargan automáticamente en `models/` la primera vez que ejecutás el programa.

---

## 🕹️ Controles

### Flappy Bird

| Control | Acción |
|---------|--------|
| **Mover la cabeza arriba/abajo** | Controlar la altura del pájaro |
| Cualquier tecla | Comenzar partida (desde el menú) |
| `R` | Reiniciar después de game over |
| `ESC` | Salir del juego |

### Detector de Gestos

| Control | Acción |
|---------|--------|
| 🐭 Manos en las orejas | Detecta gesto "Topo Gigio" |
| ⭐ Índice en la boca | Detecta gesto "Dybala" |
| `ESC` | Salir |

---

## 🧠 Cómo Funciona

### Control del Pájaro

```python
# La posición Y de la nariz se mapea a la pantalla
target_y = nose_y_norm * ALTO

# Suavizado exponencial independiente del framerate
factor = 1.0 - (1.0 - 0.3) ** (dt * 60.0)
pajarito_y += (target_y - pajarito_y) * factor
```

El pájaro sigue la nariz con un factor de suavizado de 0.3, normalizado a 60 FPS de referencia usando delta time.

### Dificultad Progresiva

| Tiempo | Velocidad | Gap (espacio) | Intervalo entre pipes |
|--------|-----------|---------------|----------------------|
| 0-15s | 300 px/s | 280 px | 2000 ms |
| 15-27s | 420 px/s | 280 px | 2000 ms |
| 27-39s | 540 px/s | 260 px | 1900 ms |
| Score ≥ 50 | +240 px/s extra | -30 px extra | -200 ms extra |

### Detección de Gestos

- **Face Mesh**: 468 landmarks. Puntos clave: nariz (1), boca (13), orejas (234, 454)
- **Hand Landmarks**: 21 landmarks por mano. Punto clave: punta del índice (8)
- Los gestos se detectan midiendo distancias normalizadas entre puntos de la mano y la cara

---

## 📁 Estructura del Proyecto

```
Flappy Bird/
├── src/                          # Código fuente
│   ├── __init__.py
│   ├── game.py                   # Juego Flappy Bird
│   └── gesture_detector.py       # Detector de gestos
├── assets/                       # Recursos visuales
│   ├── yellowbird-*.png          # Sprites del pájaro
│   ├── pipe-green.png            # Tubería
│   ├── base.png                  # Suelo
│   ├── 0.png – 9.png             # Números del puntaje
│   ├── Menu.png                  # Pantalla de menú
│   ├── riquelme.jpg              # Foto para gesto Topo Gigio
│   └── dybala.jpeg               # Foto para gesto Dybala
├── models/                       # Modelos ML (se descargan solos)
├── data/                         # Datos de runtime (high score)
├── main.py                       # Punto de entrada
├── requirements.txt              # Dependencias congeladas
├── .gitignore                    # Filtro de archivos para Git
└── README.md                     # Este archivo
```

---

## 📦 Dependencias

| Librería | Versión | Uso |
|----------|---------|-----|
| `pygame` | 2.6.1 | Ventana de juego, renderizado, sprites |
| `opencv-python` | 4.13.0 | Captura y procesamiento de cámara |
| `mediapipe` | 0.10.35 | Face Mesh y Hand Landmarker |
| `numpy` | 2.0.2 | Manipulación de arrays para imágenes |
| `Pillow` | 11.3.0 | Renderizado de emojis (solo gestos) |

---

## 📝 Notas Técnicas

- **macOS + Continuity Camera**: El juego prioriza la webcam integrada del MacBook sobre la cámara del iPhone. Si detecta la cámara equivocada, probá desactivar Continuity Camera en *Configuración del Sistema → General → AirDrop y Handoff*.
- **Apple Silicon (M1/M2/M3/M4)**: MediaPipe usa aceleración Metal automáticamente.
- Los modelos `.task` se descargan una sola vez (~11 MB total) de Google Storage.
- Si falta alguna textura en `assets/`, el juego dibuja formas de colores como fallback.
- El high score se guarda automáticamente en `data/high_score.json`.

---

## 👥 Autores

Desarrollado como proyecto académico.

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.
