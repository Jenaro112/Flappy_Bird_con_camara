#!/usr/bin/env python3
"""
Punto de entrada principal — Flappy Bird con Control de Cámara.

Uso:
    python main.py              Ejecutar el juego Flappy Bird
    python main.py gestures     Ejecutar el detector de gestos
"""
import sys
import os
import subprocess

# Directorio raíz del proyecto
_ROOT = os.path.dirname(os.path.abspath(__file__))

# Scripts disponibles
_SCRIPTS = {
    "game":     os.path.join(_ROOT, "src", "game.py"),
    "gestures": os.path.join(_ROOT, "src", "gesture_detector.py"),
}

if __name__ == "__main__":
    # Determinar qué script ejecutar
    target = "game"
    if len(sys.argv) > 1 and sys.argv[1] in ("gestures", "gesture", "gestos"):
        target = "gestures"

    script = _SCRIPTS[target]
    if not os.path.exists(script):
        print(f"ERROR: No se encontró {script}")
        sys.exit(1)

    print(f"Ejecutando: {os.path.basename(script)}")
    sys.exit(subprocess.call([sys.executable, script]))
