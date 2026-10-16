"""
Detección de manos con MediaPipe y lógica de "pulsación" de cada dedo.

Flujo por fotograma:
    imagen -> MediaPipe (21 puntos 3D por mano) -> flexión de cada dedo (grados)
           -> filtro One Euro -> comparación con la línea base -> evento "nota"
"""

import math
import urllib.request
from dataclasses import dataclass

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import config

_CADENAS = np.array(config.CADENAS_DEDOS)  # (5 dedos, 5 puntos)


@dataclass
class ManoDetectada:
    puntos: np.ndarray   # (21, 2) coordenadas normalizadas 0..1 en la imagen
    mundo: np.ndarray    # (21, 3) coordenadas 3D reales en metros (centro de la mano)
    etiqueta: str        # "Left" / "Right" según MediaPipe
    confianza: float


def asegurar_modelo(ruta=config.RUTA_MODELO, url=config.URL_MODELO):
    """Descarga el modelo preentrenado de Google la primera vez (~7.8 MB)."""
    if ruta.exists() and ruta.stat().st_size > 1_000_000:
        return ruta
    ruta.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando el modelo de manos desde {url} ...")
    temporal = ruta.with_suffix(".descargando")
    urllib.request.urlretrieve(url, temporal)
    temporal.replace(ruta)
    print("Modelo descargado en", ruta)
    return ruta


class DetectorManos:
    """Envoltura del HandLandmarker de MediaPipe (API Tasks, modo video)."""

    def __init__(self, num_manos=config.NUM_MANOS):
        modelo = asegurar_modelo()
        opciones = vision.HandLandmarkerOptions(
            # Se pasa el modelo como bytes para evitar problemas con rutas en Windows
            base_options=mp_python.BaseOptions(model_asset_buffer=modelo.read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_manos,
            min_hand_detection_confidence=config.CONF_DETECCION,
            min_hand_presence_confidence=config.CONF_PRESENCIA,
            min_tracking_confidence=config.CONF_SEGUIMIENTO,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(opciones)
        self._ultimo_ms = -1

    def detectar(self, imagen_rgb: np.ndarray, tiempo_ms: float) -> list:
        # En modo VIDEO las marcas de tiempo deben ser estrictamente crecientes
        tiempo_ms = max(int(tiempo_ms), self._ultimo_ms + 1)
        self._ultimo_ms = tiempo_ms
        imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagen_rgb)
        resultado = self._landmarker.detect_for_video(imagen, tiempo_ms)

        manos = []
        for lm2d, lm3d, lado in zip(resultado.hand_landmarks,
                                    resultado.hand_world_landmarks,
                                    resultado.handedness):
            puntos = np.array([[p.x, p.y] for p in lm2d], dtype=np.float32)
            mundo = np.array([[p.x, p.y, p.z] for p in lm3d], dtype=np.float32)
            manos.append(ManoDetectada(puntos, mundo, lado[0].category_name, lado[0].score))
        return manos

    def cerrar(self):
        self._landmarker.close()


def flexion_dedos(mundo: np.ndarray) -> np.ndarray:
    """
    Flexión de los 5 dedos en grados (0 = dedo recto).

    Para cada articulación se calcula el ángulo entre el hueso anterior y el
    siguiente con el producto punto: cos(theta) = (u . v) / (|u| |v|).
    La flexión del dedo es la suma de los ángulos de sus 3 articulaciones.
    Al usar coordenadas 3D reales no depende de la distancia a la cámara ni de
    la rotación de la mano: mover la mano completa NO cambia la flexión.
    """
    p = mundo[_CADENAS]                     # (5, 5, 3)
    huesos = np.diff(p, axis=1)             # (5, 4, 3) vectores entre articulaciones
    huesos /= np.linalg.norm(huesos, axis=2, keepdims=True) + 1e-9
    cosenos = np.sum(huesos[:, :-1] * huesos[:, 1:], axis=2)   # (5, 3)
    return np.degrees(np.arccos(np.clip(cosenos, -1.0, 1.0))).sum(axis=1)


class FiltroUnEuro:
    """
    Filtro "One Euro" (Casiez et al., 2012), estándar en seguimiento para RV/RA.
    Suaviza mucho cuando la señal está quieta (quita el temblor) y poco cuando
    cambia rápido (no agrega retraso a una pulsación). Funciona con vectores.
    """

    def __init__(self, min_corte, beta, corte_derivada):
        self.min_corte = min_corte
        self.beta = beta
        self.corte_derivada = corte_derivada
        self.valor = None
        self.derivada = None
        self.t = None

    @staticmethod
    def _alfa(dt, corte):
        tau = 1.0 / (2.0 * math.pi * corte)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.valor is None:
            self.valor, self.derivada, self.t = x.copy(), np.zeros_like(x), t
            return self.valor
        dt = max(t - self.t, 1e-3)
        a_d = self._alfa(dt, self.corte_derivada)
        self.derivada = a_d * (x - self.valor) / dt + (1 - a_d) * self.derivada
        a = self._alfa(dt, self.min_corte + self.beta * np.abs(self.derivada))
        self.valor = a * x + (1 - a) * self.valor
        self.t = t
        return self.valor


class EstadoMano:
    """
    Memoria de una mano entre fotogramas: filtra la flexión de cada dedo y
    detecta "golpes" (pulsaciones) con histéresis:

    - Dedo libre: se sigue el "valle" (punto más estirado reciente). El valle baja
      al instante y sube lento, así una postura que cambia despacio no suena.
      Si el dedo se dobla más de `umbral` grados sobre el valle, con velocidad
      suficiente -> NOTA.
    - Dedo presionado: se sigue el pico de flexión. Cuando el dedo retrocede
      `umbral * FACTOR_LIBERACION` grados desde el pico, queda listo otra vez
      (permite repetir la nota sin estirar el dedo por completo).
    """

    def __init__(self):
        self.reiniciar()

    def reiniciar(self):
        self.filtro = FiltroUnEuro(config.FILTRO_MIN_CORTE, config.FILTRO_BETA,
                                   config.FILTRO_CORTE_DERIVADA)
        self.flexion = np.zeros(5)        # flexión suavizada (grados)
        self.valle = None                 # referencia de reposo de cada dedo
        self.pico = np.zeros(5)
        self.relativa = np.zeros(5)       # flexión - valle (lo que se compara con el umbral)
        self.velocidad = np.zeros(5)      # grados por segundo
        self.presionado = np.zeros(5, dtype=bool)
        self.ultimo_disparo = np.full(5, -1e9)
        self.frames = 0
        self.t = None
        self.visto = None                 # último instante en que se vio la mano

    def perder(self, t):
        """La mano no apareció en este fotograma: se reinicia tras un tiempo de gracia."""
        if self.visto is not None and t - self.visto > config.TIEMPO_GRACIA:
            self.reiniciar()

    def actualizar(self, flexion_cruda: np.ndarray, t: float, umbrales: np.ndarray) -> list:
        """Devuelve la lista de eventos [(dedo, velocidad)] de dedos que acaban de presionar."""
        dt = 0.0 if self.t is None else max(t - self.t, 1e-3)
        self.t = self.visto = t
        self.flexion = self.filtro(flexion_cruda.astype(np.float64), t)
        self.velocidad = self.filtro.derivada
        if self.valle is None:
            self.valle = self.flexion.copy()

        k_sube = 1.0 - math.exp(-dt / config.TAU_VALLE)
        calentando = self.frames < config.FRAMES_CALENTAMIENTO
        self.frames += 1

        eventos = []
        for d in range(5):
            f = self.flexion[d]
            if self.presionado[d]:
                self.pico[d] = max(self.pico[d], f)
                if f < self.pico[d] - umbrales[d] * config.FACTOR_LIBERACION:
                    self.presionado[d] = False
                    self.valle[d] = f
                continue

            if f < self.valle[d]:
                self.valle[d] = f
            else:
                self.valle[d] += k_sube * (f - self.valle[d])

            if (not calentando
                    and f - self.valle[d] > umbrales[d]
                    and self.velocidad[d] > config.VELOCIDAD_MINIMA
                    and t - self.ultimo_disparo[d] > config.REBOTE):
                self.presionado[d] = True
                self.pico[d] = f
                self.ultimo_disparo[d] = t
                eventos.append((d, float(self.velocidad[d])))

        self.relativa = self.flexion - self.valle
        return eventos


def asignar_manos(manos: list, munecas_previas: dict) -> dict:
    """
    Decide qué mano es la "izquierda" (notas graves) y cuál la "derecha" (agudas).
    - Con dos manos: por posición en pantalla (más robusto que la etiqueta).
    - Con una mano: se mantiene el lado que tenía en el fotograma anterior si está
      cerca; si no, se usa la etiqueta Left/Right de MediaPipe.
    """
    if len(manos) >= 2:
        a, b = sorted(manos[:2], key=lambda m: m.puntos[0, 0])
        return {"izq": a, "der": b}
    if len(manos) == 1:
        mano = manos[0]
        cercano, distancia = None, float("inf")
        for lado, muneca in munecas_previas.items():
            d = float(np.linalg.norm(mano.puntos[0] - muneca))
            if d < distancia:
                cercano, distancia = lado, d
        if cercano is not None and distancia < 0.15:
            return {cercano: mano}
        return {"izq" if mano.etiqueta == "Left" else "der": mano}
    return {}
