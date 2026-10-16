"""
Configuración general del Piano Virtual con Manos.

Aquí están todos los parámetros ajustables (umbrales, colores, escalas, etc.)
para que el resto del código no tenga "números mágicos".
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent
RUTA_MODELO = RAIZ / "modelos" / "hand_landmarker.task"
URL_MODELO = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
CARPETA_GRABACIONES = RAIZ / "grabaciones"

# ----------------------------------------------------------------------------
# Cámara y ventana
# ----------------------------------------------------------------------------
ANCHO = 1280
ALTO = 720
NOMBRE_VENTANA = "Piano Virtual con Manos"

# ----------------------------------------------------------------------------
# Modelo de IA (MediaPipe Hand Landmarker, ya entrenado por Google)
# ----------------------------------------------------------------------------
NUM_MANOS = 2
CONF_DETECCION = 0.6     # confianza mínima para detectar una mano nueva
CONF_PRESENCIA = 0.5     # confianza mínima de que la mano sigue presente
CONF_SEGUIMIENTO = 0.5   # confianza mínima para seguir la mano entre fotogramas

# Índices de los 21 puntos clave (landmarks) de MediaPipe:
#   0 = muñeca | pulgar 1-4 | índice 5-8 | medio 9-12 | anular 13-16 | meñique 17-20
# Cada dedo se describe como una cadena muñeca -> ... -> punta.
CADENAS_DEDOS = (
    (0, 1, 2, 3, 4),      # pulgar
    (0, 5, 6, 7, 8),      # índice
    (0, 9, 10, 11, 12),   # medio
    (0, 13, 14, 15, 16),  # anular
    (0, 17, 18, 19, 20),  # meñique
)
PUNTAS = (4, 8, 12, 16, 20)
NOMBRES_DEDOS = ("Pulgar", "Índice", "Medio", "Anular", "Meñique")
CONEXIONES_MANO = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)

# ----------------------------------------------------------------------------
# Detección de "pulsación" de cada dedo
# ----------------------------------------------------------------------------
# Flexión = suma de los ángulos de doblez de las articulaciones del dedo (en 3D).
# Suena una nota cuando el dedo se dobla UMBRAL_PRESION grados sobre su "valle"
# (su postura de reposo, que se aprende sola y se adapta a cada persona).
UMBRAL_PRESION = (26.0, 30.0, 30.0, 30.0, 30.0)   # pulgar, índice, medio, anular, meñique
FACTOR_LIBERACION = 0.4     # histéresis: queda listo al retroceder umbral*factor desde el pico
VELOCIDAD_MINIMA = 60.0     # grados/s: doblar el dedo muy despacio no toca notas
TAU_VALLE = 1.0             # s: qué tan rápido el valle se adapta a una nueva postura
SENSIBILIDAD_MIN, SENSIBILIDAD_MAX = 0.5, 2.0

# Filtro One Euro (suavizado adaptativo de la flexión)
FILTRO_MIN_CORTE = 1.5      # Hz: suavizado en reposo (menor = más suave)
FILTRO_BETA = 0.02          # cuánto se reduce el suavizado al moverse rápido
FILTRO_CORTE_DERIVADA = 1.0

FRAMES_CALENTAMIENTO = 4    # fotogramas sin disparar cuando aparece una mano
TIEMPO_GRACIA = 0.35        # s que se conserva el estado si la mano se pierde un instante
REBOTE = 0.09               # s mínimos entre dos notas del mismo dedo

# Velocidad del dedo (grados/s) -> intensidad del sonido (como un teclado MIDI)
VEL_SUAVE = 100.0
VEL_FUERTE = 500.0
INTENSIDAD_MIN = 0.35

# ----------------------------------------------------------------------------
# Música
# ----------------------------------------------------------------------------
NOMBRES_NOTAS = ("Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si")

# Escalas (intervalos en semitonos desde Do)
ESCALAS = (
    ("Do mayor", (0, 2, 4, 5, 7, 9, 11)),
    ("Pentatónica", (0, 2, 4, 7, 9)),
    ("Do menor", (0, 2, 3, 5, 7, 8, 10)),
    ("Blues", (0, 3, 5, 6, 7, 10)),
)
OCTAVA_INICIAL = 4
OCTAVA_MIN, OCTAVA_MAX = 2, 6


def nombre_nota(midi: int) -> str:
    """Convierte un número MIDI en su nombre en solfeo, ej. 60 -> 'Do4'."""
    return f"{NOMBRES_NOTAS[midi % 12]}{midi // 12 - 1}"


def notas_para_dedos(intervalos: tuple, octava: int) -> list:
    """
    Asigna una nota a cada uno de los 10 dedos, de izquierda a derecha en pantalla:
    meñique, anular, medio, índice, pulgar (mano izquierda) y
    pulgar, índice, medio, anular, meñique (mano derecha).
    """
    do = 12 * (octava + 1)  # número MIDI del Do de esa octava (Do4 = 60)
    n = len(intervalos)
    return [do + 12 * (g // n) + intervalos[g % n] for g in range(10)]


def posicion_dedo(lado: str, dedo: int) -> int:
    """Posición 0..9 (izquierda -> derecha) de un dedo. dedo: 0=pulgar ... 4=meñique."""
    return 4 - dedo if lado == "izq" else 5 + dedo


def lado_y_dedo(posicion: int) -> tuple:
    """Inverso de posicion_dedo: 0..9 -> ('izq' o 'der', dedo)."""
    return ("izq", 4 - posicion) if posicion < 5 else ("der", posicion - 5)


# ----------------------------------------------------------------------------
# Colores (formato BGR de OpenCV)
# ----------------------------------------------------------------------------
COLORES_DEDOS = (
    (107, 107, 255),  # pulgar   - coral
    (77, 169, 255),   # índice   - naranja
    (59, 212, 255),   # medio    - amarillo
    (124, 219, 105),  # anular   - verde
    (247, 171, 77),   # meñique  - azul
)
COLOR_PANEL = (28, 22, 18)
COLOR_ACENTO = (238, 211, 34)
COLOR_OK = (102, 207, 81)
COLOR_ERROR = (90, 90, 250)
COLOR_TEXTO = (245, 245, 245)
COLOR_TEXTO_SUAVE = (185, 185, 185)
