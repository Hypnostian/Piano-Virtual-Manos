"""
Modo canción: el piano te guía nota por nota (qué dedo usar) y mide tu precisión.

Las canciones en Do mayor (octava 4) usan las notas normales de los dedos:
    mano izquierda: Do4 Re4 Mi4 Fa4 Sol4  (meñique -> pulgar)
    mano derecha:   La4 Si4 Do5 Re5 Mi5   (pulgar -> meñique)

Si una canción necesita otras notas (sostenidos, bemoles u otra octava), las notas de los
dedos cambian SOLO para esa canción. Si usa más de 10 notas distintas, se divide en tramos
y las notas de los dedos se alternan entre frases, justo cuando hace falta.

Agregar una canción: escribe sus notas en un texto con cualquiera de estos formatos (se
pueden mezclar) y separa las frases con "|" o con saltos de línea:
    Solfeo:   "Mi5 Re#5 Mi5 Sib4 Do5"
    Letras:   "E5 D#5 E5 Bb4 C5"
    Kalimba:  "3° 2° 3° 7 1°"   (1=Do ... 7=Si, 1 = Do4; cada ° sube una octava)
Los acordes entre paréntesis, ej. "(135)", se toman como su nota más aguda (la melodía).
"""

import re
from collections import namedtuple

import config

DO4, RE4, MI4, FA4, SOL4, LA4, SI4, DO5, RE5, MI5 = 60, 62, 64, 65, 67, 69, 71, 72, 74, 76
NOTAS_NORMALES = config.notas_para_dedos(config.ESCALAS[0][1], 4)   # Do mayor, octava 4

Tramo = namedtuple("Tramo", "inicio fin asignacion")   # asignacion None = notas normales


class Melodia(list):
    """Lista de notas MIDI que además recuerda dónde empieza cada frase."""

    def __init__(self, notas=(), frases=(0,)):
        super().__init__(notas)
        self.frases = list(frases)


# ----------------------------------------------------------------------------
# Lectura de notas escritas como texto
# ----------------------------------------------------------------------------
_SOLFEO = {"do": 0, "re": 2, "mi": 4, "fa": 5, "sol": 7, "la": 9, "si": 11}
_LETRAS = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
_ALTERACION = {"": 0, "#": 1, "♯": 1, "b": -1, "♭": -1}
_KALIMBA = {"1": 0, "2": 2, "3": 4, "4": 5, "5": 7, "6": 9, "7": 11}
_PATRON = re.compile(
    r"(?P<solfeo>do|re|mi|fa|sol|la|si)(?P<alt_s>[#♯b♭]?)(?P<oct_s>\d)"
    r"|(?P<letra>[a-g])(?P<alt_l>[#♯b♭]?)(?P<oct_l>\d)"
    r"|(?P<kalimba>[1-7])(?P<arriba>[°º*']*)(?P<abajo>̣*)"
    r"|(?P<frase>[|\n])|(?P<abre>\()|(?P<cierra>\))",
    re.IGNORECASE,
)


def notas(texto: str) -> Melodia:
    """Convierte un texto con notas (solfeo, letras o números de kalimba) en una Melodia."""
    resultado, frases, acorde = [], [0], None
    for m in _PATRON.finditer(texto):
        if m["frase"]:
            frases.append(len(resultado))
        elif m["abre"]:
            acorde = []
        elif m["cierra"]:
            if acorde:
                resultado.append(max(acorde))
            acorde = None
        else:
            if m["solfeo"]:
                nota = _SOLFEO[m["solfeo"].lower()] + _ALTERACION[m["alt_s"].lower()] + 12 * (int(m["oct_s"]) + 1)
            elif m["letra"]:
                nota = _LETRAS[m["letra"].lower()] + _ALTERACION[m["alt_l"].lower()] + 12 * (int(m["oct_l"]) + 1)
            else:
                nota = DO4 + _KALIMBA[m["kalimba"]] + 12 * (len(m["arriba"]) - len(m["abajo"]))
            (acorde if acorde is not None else resultado).append(nota)
    frases = sorted({f for f in frases if f < len(resultado)} | {0})
    return Melodia(resultado, frases)


# ----------------------------------------------------------------------------
# Canciones
# ----------------------------------------------------------------------------
_TODAS = (
    ("Escala de Do (calentamiento)",
     [DO4, RE4, MI4, FA4, SOL4, LA4, SI4, DO5, RE5, MI5,
      RE5, DO5, SI4, LA4, SOL4, FA4, MI4, RE4, DO4]),
    ("Himno de la Alegría (Beethoven)",
     [MI4, MI4, FA4, SOL4, SOL4, FA4, MI4, RE4, DO4, DO4, RE4, MI4, MI4, RE4, RE4,
      MI4, MI4, FA4, SOL4, SOL4, FA4, MI4, RE4, DO4, DO4, RE4, MI4, RE4, DO4, DO4]),
    ("Estrellita, ¿dónde estás?",
     [DO4, DO4, SOL4, SOL4, LA4, LA4, SOL4, FA4, FA4, MI4, MI4, RE4, RE4, DO4,
      SOL4, SOL4, FA4, FA4, MI4, MI4, RE4, SOL4, SOL4, FA4, FA4, MI4, MI4, RE4,
      DO4, DO4, SOL4, SOL4, LA4, LA4, SOL4, FA4, FA4, MI4, MI4, RE4, RE4, DO4]),
    ("Jingle Bells",
     [MI4, MI4, MI4, MI4, MI4, MI4, MI4, SOL4, DO4, RE4, MI4,
      FA4, FA4, FA4, FA4, FA4, MI4, MI4, MI4, MI4, RE4, RE4, MI4, RE4, SOL4,
      MI4, MI4, MI4, MI4, MI4, MI4, MI4, SOL4, DO4, RE4, MI4,
      FA4, FA4, FA4, FA4, FA4, MI4, MI4, MI4, SOL4, SOL4, FA4, RE4, DO4]),
    # Necesita Re# y Sol# y usa 12 notas distintas: las notas de los dedos se alternan
    # automáticamente al empezar la tercera frase.
    ("Para Elisa (Beethoven)", notas("""
        Mi5 Re#5 Mi5 Re#5 Mi5 Si4 Re5 Do5 La4  Do4 Mi4 La4 Si4  Mi4 Sol#4 Si4 Do5
        Mi4 Mi5 Re#5 Mi5 Re#5 Mi5 Si4 Re5 Do5 La4  Do4 Mi4 La4 Si4  Mi4 Do5 Si4 La4
        Si4 Do5 Re5 Mi5  Sol4 Fa5 Mi5 Re5  Fa4 Mi5 Re5 Do5  Mi4 Re5 Do5 Si4
        Mi5 Re#5 Mi5 Re#5 Mi5 Si4 Re5 Do5 La4
    """)),
    # Pega aquí la melodía de El Castillo Ambulante (Merry-Go-Round of Life) en cualquier
    # formato de los de arriba, una frase por línea. Mientras esté vacía no aparece.
    ("El Castillo Ambulante (Joe Hisaishi)", notas("""
    """)),
)
CANCIONES = tuple(cancion for cancion in _TODAS if cancion[1])   # las vacías no se muestran


# ----------------------------------------------------------------------------
# Notas de los dedos para cada canción
# ----------------------------------------------------------------------------
def asignacion_dedos(necesarias) -> list:
    """
    10 notas de grave a aguda (una por dedo) que incluyen todas las necesarias (máximo 10).
    Si sobran dedos, se completan con las notas naturales vecinas, alternando arriba y abajo.
    """
    asignadas = set(necesarias)
    arriba, abajo, subir = max(asignadas), min(asignadas), True
    while len(asignadas) < 10:
        if subir:
            arriba += 1
            while arriba % 12 not in config.ESCALAS[0][1]:
                arriba += 1
            asignadas.add(arriba)
        else:
            abajo -= 1
            while abajo % 12 not in config.ESCALAS[0][1]:
                abajo -= 1
            asignadas.add(abajo)
        subir = not subir
    return sorted(asignadas)


def calcular_tramos(notas_cancion: list, frases=(0,)) -> list:
    """
    Divide la canción en tramos cuyas notas caben en los 10 dedos.
    - Si toda la canción cabe en las notas normales: un solo tramo sin cambios (None).
    - Si no: se juntan frases completas mientras quepan (máx. 10 notas distintas), así las
      notas de los dedos cambian entre frases y no en medio de una. Una frase con más de
      10 notas distintas se corta donde haga falta.
    """
    if set(notas_cancion) <= set(NOTAS_NORMALES):
        return [Tramo(0, len(notas_cancion), None)]
    limites = sorted(set(frases) | {0}) + [len(notas_cancion)]
    tramos, inicio, actuales = [], 0, set()
    for a, b in zip(limites, limites[1:]):
        frase = set(notas_cancion[a:b])
        if len(actuales | frase) <= 10:
            actuales |= frase
            continue
        if actuales:
            tramos.append(Tramo(inicio, a, asignacion_dedos(actuales)))
            inicio, actuales = a, set()
        for i in range(a, b):
            if notas_cancion[i] not in actuales and len(actuales) == 10:
                tramos.append(Tramo(inicio, i, asignacion_dedos(actuales)))
                inicio, actuales = i, set()
            actuales.add(notas_cancion[i])
    tramos.append(Tramo(inicio, len(notas_cancion), asignacion_dedos(actuales)))
    return tramos


class ModoCancion:
    def __init__(self, indice: int):
        self.indice_cancion = indice
        self.nombre, self.notas = CANCIONES[indice]
        self.tramos = calcular_tramos(self.notas, getattr(self.notas, "frases", (0,)))
        self.reiniciar()

    def reiniciar(self):
        self.posicion = 0
        self.aciertos = 0
        self.errores = 0
        self.t_inicio = None
        self.t_fin = None

    @property
    def terminada(self) -> bool:
        return self.posicion >= len(self.notas)

    @property
    def esperada(self):
        return None if self.terminada else self.notas[self.posicion]

    @property
    def precision(self) -> float:
        total = self.aciertos + self.errores
        return self.aciertos / total if total else 1.0

    @property
    def necesita_otras_notas(self) -> bool:
        return self.tramos[0].asignacion is not None

    def tramo_actual(self) -> int:
        """Número (0, 1, ...) del tramo en el que va la canción."""
        posicion = min(self.posicion, len(self.notas) - 1)
        return next(i for i, tramo in enumerate(self.tramos) if posicion < tramo.fin)

    def asignacion_actual(self):
        """Notas de los dedos para el tramo actual, o None si sirven las notas normales."""
        return self.tramos[self.tramo_actual()].asignacion

    def proximas(self, cantidad: int = 8) -> list:
        return self.notas[self.posicion:self.posicion + cantidad]

    def registrar(self, notas_tocadas: list, t: float):
        """
        Recibe las notas tocadas en un fotograma. Devuelve True (acierto),
        False (error) o None (nada que evaluar). Si en el mismo instante suena la
        nota correcta junto con otra (dedo vecino que se movió), cuenta como acierto.
        """
        if self.terminada or not notas_tocadas:
            return None
        if self.t_inicio is None:
            self.t_inicio = t
        if self.esperada in notas_tocadas:
            self.posicion += 1
            self.aciertos += 1
            if self.terminada:
                self.t_fin = t
            return True
        self.errores += 1
        return False
