"""
Sonido del piano: síntesis aditiva (no necesita archivos de audio) y grabación a WAV.
"""

import os
import threading
import wave
from datetime import datetime

import numpy as np

import config

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

FS = 44100  # muestras por segundo


def frecuencia(midi: int) -> float:
    """Frecuencia en Hz de una nota MIDI (La4 = 69 = 440 Hz)."""
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def sintetizar_nota(midi: int, fs: int = FS) -> np.ndarray:
    """
    Genera una nota con timbre de piano mediante síntesis aditiva:
    - Suma de armónicos k*f0 con leve inarmonicidad (las cuerdas reales son rígidas).
    - El martillo golpea la cuerda a 1/7 de su largo: se atenúan los armónicos 7, 14...
    - Cada armónico decae exponencialmente (los agudos se apagan antes) con
      "doble caída": un ataque brillante y una cola más larga.
    - Un ruido muy corto al inicio imita el golpe del martillo.
    Devuelve un arreglo float32 estéreo (n, 2) con valores en [-1, 1].
    """
    f0 = frecuencia(midi)
    duracion = float(np.clip(3.2 - 0.045 * (midi - 48), 1.2, 3.2))  # los graves suenan más
    t = np.arange(int(duracion * fs)) / fs
    onda = np.zeros_like(t)
    tasa = 1.1 + f0 / 380.0
    # El armónico k decae con exp(-tasa * (1 + 0.55 (k-1)) * t). En lugar de llamar
    # a exp() para cada armónico, la envolvente se obtiene multiplicando por un "paso".
    rapida, paso_rapida = np.exp(-tasa * t), np.exp(-0.55 * tasa * t)
    lenta, paso_lenta = np.exp(-0.3 * tasa * t), np.exp(-0.165 * tasa * t)
    for k in range(1, 20):
        fk = k * f0 * np.sqrt(1.0 + 0.0004 * k * k)
        if fk > 0.45 * fs:
            break
        amplitud = abs(np.sin(np.pi * k / 7.0)) / k ** 1.05
        amplitud *= np.exp(-(k - 1) * f0 / 4000.0)
        onda += amplitud * (0.75 * rapida + 0.25 * lenta) * np.sin(2.0 * np.pi * fk * t + 0.7 * k)
        rapida *= paso_rapida
        lenta *= paso_lenta
    onda /= np.abs(onda).max()

    # Golpe del martillo: ruido filtrado de pocos milisegundos
    n_golpe = int(0.03 * fs)
    ruido = np.random.default_rng(midi).standard_normal(n_golpe)
    ruido = np.convolve(ruido, np.ones(6) / 6.0, mode="same")
    onda[:n_golpe] += 0.12 * ruido * np.exp(-np.arange(n_golpe) / (0.004 * fs))

    # Inicio y final suaves para evitar "clics"
    n_ataque, n_final = int(0.003 * fs), int(0.08 * fs)
    onda[:n_ataque] *= np.linspace(0.0, 1.0, n_ataque)
    onda[-n_final:] *= np.linspace(1.0, 0.0, n_final)

    # Paneo estéreo: graves a la izquierda y agudos a la derecha, como al piano
    paneo = float(np.clip((midi - 66) / 30.0, -0.6, 0.6))
    angulo = (paneo + 1.0) * np.pi / 4.0
    estereo = np.stack([onda * np.cos(angulo), onda * np.sin(angulo)], axis=1)
    return (0.9 * estereo / np.abs(estereo).max()).astype(np.float32)


class Sintetizador:
    """Reproduce las notas con pygame (polifonía de 32 voces) y guarda en caché las ondas."""

    def __init__(self, activo=True, volumen=0.75):
        self.fs = FS
        self.volumen = volumen
        self.silenciado = False
        self._ondas = {}
        self._sonidos = {}
        self._canales = {}
        self._pygame = None
        if activo:
            try:
                import pygame
                # allowedchanges=0: SDL convierte al formato de la tarjeta de sonido
                pygame.mixer.init(frequency=FS, size=-16, channels=2, buffer=512, allowedchanges=0)
                pygame.mixer.set_num_channels(32)
                self._pygame = pygame
            except Exception as error:  # sin tarjeta de sonido, etc.
                print(f"Aviso: no se pudo iniciar el audio ({error}). Se continúa sin sonido.")

    @property
    def con_audio(self) -> bool:
        return self._pygame is not None

    def onda(self, midi: int) -> np.ndarray:
        if midi not in self._ondas:
            self._ondas[midi] = sintetizar_nota(midi, self.fs)
        return self._ondas[midi]

    def _sonido(self, midi: int):
        if midi not in self._sonidos:
            datos = np.ascontiguousarray((self.onda(midi) * 32767).astype(np.int16))
            self._sonidos[midi] = self._pygame.sndarray.make_sound(datos)
        return self._sonidos[midi]

    def precargar(self, notas):
        """Genera por adelantado las notas que se van a usar (evita pausas al tocar)."""
        for midi in notas:
            if self.con_audio:
                self._sonido(midi)
            else:
                self.onda(midi)

    def precalentar(self, desde=36, hasta=106):
        """Sintetiza en segundo plano todas las notas posibles (Do2 a La7), así cambiar
        de escala u octava es instantáneo."""
        def trabajo():
            for midi in range(desde, hasta):
                self.onda(midi)
        threading.Thread(target=trabajo, daemon=True).start()

    def tocar(self, midi: int, intensidad: float = 1.0):
        if not self.con_audio or self.silenciado:
            return
        sonido = self._sonido(midi)
        anterior = self._canales.get(midi)
        if anterior is not None and anterior.get_busy() and anterior.get_sound() == sonido:
            anterior.fadeout(70)  # al repetir una nota, la anterior se apaga suavemente
        canal = sonido.play()
        if canal is not None:
            canal.set_volume(self.volumen * intensidad)
            self._canales[midi] = canal

    def cerrar(self):
        if self.con_audio:
            self._pygame.mixer.quit()


class Grabadora:
    """Registra lo que se toca (tiempo, nota, intensidad) y lo exporta a un archivo WAV."""

    def __init__(self):
        self.eventos = []
        self.inicio = None

    @property
    def grabando(self) -> bool:
        return self.inicio is not None

    def duracion(self, t: float) -> float:
        return t - self.inicio if self.grabando else 0.0

    def iniciar(self, t: float):
        self.eventos, self.inicio = [], t

    def registrar(self, t: float, midi: int, intensidad: float):
        if self.grabando:
            self.eventos.append((t - self.inicio, midi, intensidad))

    def detener(self, sintetizador: Sintetizador, carpeta=config.CARPETA_GRABACIONES):
        """Detiene la grabación y guarda el WAV. Devuelve la ruta o None si no hubo notas."""
        eventos, self.inicio = self.eventos, None
        if not eventos:
            return None
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta = carpeta / f"piano_{datetime.now():%Y%m%d_%H%M%S}.wav"
        guardar_wav(ruta, mezclar(eventos, sintetizador), sintetizador.fs)
        return ruta


def mezclar(eventos: list, sintetizador: Sintetizador) -> np.ndarray:
    """Suma todas las notas en una sola pista estéreo, respetando sus tiempos."""
    fs = sintetizador.fs
    eventos = sorted(eventos)
    t0 = eventos[0][0] - 0.1
    largo = max(int((t - t0) * fs) + len(sintetizador.onda(m)) for t, m, _ in eventos)
    pista = np.zeros((largo, 2), dtype=np.float32)
    n_apagado = int(0.07 * fs)
    for i, (t, midi, intensidad) in enumerate(eventos):
        nota = sintetizador.onda(midi) * (sintetizador.volumen * intensidad)
        # Igual que en vivo: si la nota se repite, la anterior se apaga
        repeticion = next((t2 for t2, m2, _ in eventos[i + 1:] if m2 == midi), None)
        if repeticion is not None:
            fin = int((repeticion - t) * fs) + n_apagado
            if fin < len(nota):
                nota = nota[:fin].copy()
                nota[-n_apagado:] *= np.linspace(1.0, 0.0, n_apagado)[:, None]
        inicio = int((t - t0) * fs)
        pista[inicio:inicio + len(nota)] += nota
    pico = np.abs(pista).max()
    if pico > 0.99:
        pista *= 0.99 / pico
    return pista


def guardar_wav(ruta, pista: np.ndarray, fs: int = FS):
    with wave.open(str(ruta), "wb") as archivo:
        archivo.setnchannels(2)
        archivo.setsampwidth(2)
        archivo.setframerate(fs)
        archivo.writeframes((pista * 32767).astype("<i2").tobytes())
