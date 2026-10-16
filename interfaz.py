"""
Dibujo de la interfaz sobre la imagen de la cámara (solo OpenCV).

Con OpenCV 5 se usa su nuevo motor de texto Unicode (tildes, ñ, símbolos ♪);
con OpenCV 4 se usa la fuente Hershey clásica quitando las tildes.
"""

import math
import unicodedata

import cv2
import numpy as np

import config

_FUENTE = cv2.FontFace("sans") if hasattr(cv2, "FontFace") else None
_TECLAS_BLANCAS = (0, 2, 4, 5, 7, 9, 11)


# ----------------------------------------------------------------------------
# Utilidades de dibujo
# ----------------------------------------------------------------------------
def _sin_tildes(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def ancho_texto(s: str, tam: float, peso: int = 400) -> int:
    tam = max(int(round(tam)), 8)
    if _FUENTE is not None:
        return cv2.getTextSize((0, 0), s, (0, 0), _FUENTE, tam, peso)[2]
    grosor = 2 if peso >= 600 else 1
    return cv2.getTextSize(_sin_tildes(s), cv2.FONT_HERSHEY_SIMPLEX, tam / 32.0, grosor)[0][0]


def texto(img, s, x, y, tam=18, color=config.COLOR_TEXTO, peso=400, alinear="izq", sombra=True):
    """Escribe texto; (x, y) es la línea base. alinear: 'izq', 'centro' o 'der'."""
    tam = max(int(round(tam)), 8)
    x, y = int(x), int(y)
    if alinear != "izq":
        ancho = ancho_texto(s, tam, peso)
        x -= ancho // 2 if alinear == "centro" else ancho
    if _FUENTE is not None:
        if sombra:
            cv2.putText(img, s, (x + 1, y + 2), (0, 0, 0), _FUENTE, tam, peso)
        cv2.putText(img, s, (x, y), color, _FUENTE, tam, peso)
        return
    s = _sin_tildes(s)
    escala, grosor = tam / 32.0, 2 if peso >= 600 else 1
    if sombra:
        cv2.putText(img, s, (x + 1, y + 2), cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), grosor + 1, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)


def mezclar_color(c1, c2, a: float):
    """Interpola entre dos colores: a=0 -> c1, a=1 -> c2."""
    a = min(max(a, 0.0), 1.0)
    return tuple(int(round(p + (q - p) * a)) for p, q in zip(c1, c2))


def rectangulo_redondeado(img, p1, p2, color, radio=10, grosor=-1):
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    r = int(max(1, min(radio, (x2 - x1) // 2, (y2 - y1) // 2)))
    if grosor < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
        return
    for a, b in (((x1 + r, y1), (x2 - r, y1)), ((x1 + r, y2), (x2 - r, y2)),
                 ((x1, y1 + r), (x1, y2 - r)), ((x2, y1 + r), (x2, y2 - r))):
        cv2.line(img, a, b, color, grosor, cv2.LINE_AA)
    for centro, angulo in (((x1 + r, y1 + r), 180), ((x2 - r, y1 + r), 270),
                           ((x2 - r, y2 - r), 0), ((x1 + r, y2 - r), 90)):
        cv2.ellipse(img, centro, (r, r), angulo, 0, 90, color, grosor, cv2.LINE_AA)


def panel(img, p1, p2, color=config.COLOR_PANEL, alfa=0.72, radio=12):
    """Rectángulo semitransparente; solo se mezcla esa región (es rápido)."""
    h, w = img.shape[:2]
    x1, y1 = max(int(p1[0]), 0), max(int(p1[1]), 0)
    x2, y2 = min(int(p2[0]), w), min(int(p2[1]), h)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return
    roi = img[y1:y2, x1:x2]
    capa = roi.copy()
    rectangulo_redondeado(capa, (0, 0), (x2 - x1 - 1, y2 - y1 - 1), color, radio)
    cv2.addWeighted(capa, alfa, roi, 1 - alfa, 0, dst=roi)


class Diseno:
    """Posiciones y tamaños de la interfaz según el tamaño de la imagen."""

    def __init__(self, ancho: int, alto: int):
        self.w, self.h = ancho, alto
        self.s = max(0.6, min(ancho / 1280.0, alto / 720.0))
        self.alto_barra = int(52 * self.s)
        margen = int(18 * self.s)
        alto_teclado = int(alto * 0.2)
        self.teclado = (margen, alto - margen - alto_teclado, ancho - margen, alto - margen)


# ----------------------------------------------------------------------------
# Mano
# ----------------------------------------------------------------------------
def _dedo_de_conexion(a: int, b: int):
    """Índice del dedo al que pertenece un hueso, o None si es de la palma."""
    if a == 0 or (a - 1) // 4 != (b - 1) // 4:
        return None
    return (b - 1) // 4


_DEDO_CONEXION = [_dedo_de_conexion(a, b) for a, b in config.CONEXIONES_MANO]


def dibujar_mano(img, d: Diseno, puntos, estado, umbrales, nombres_notas, dedo_objetivo, t):
    """Esqueleto de la mano, nota de cada dedo y un anillo que se llena al doblarlo."""
    h, w = img.shape[:2]
    s = d.s
    pts = (puntos * (w, h)).astype(int)
    for (a, b), dedo in zip(config.CONEXIONES_MANO, _DEDO_CONEXION):
        color = (225, 225, 225) if dedo is None else config.COLORES_DEDOS[dedo]
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, max(2, int(3 * s)), cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, tuple(p), max(2, int(3 * s)), (255, 255, 255), -1, cv2.LINE_AA)

    for dedo, punta in enumerate(config.PUNTAS):
        x, y = pts[punta]
        color = config.COLORES_DEDOS[dedo]
        radio = int(13 * s)
        if estado.presionado[dedo]:
            cv2.circle(img, (x, y), int(radio * 1.15), color, -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), int(radio * 1.15), (255, 255, 255), 2, cv2.LINE_AA)
        else:
            # Anillo de "carga": se llena a medida que el dedo se acerca al umbral
            avance = float(np.clip(estado.relativa[dedo] / umbrales[dedo], 0.0, 1.0))
            cv2.circle(img, (x, y), radio, (40, 40, 40), max(2, int(3 * s)), cv2.LINE_AA)
            if avance > 0.05:
                cv2.ellipse(img, (x, y), (radio, radio), -90, 0, 360 * avance, color,
                            max(2, int(4 * s)), cv2.LINE_AA)
        if dedo == dedo_objetivo:
            pulso = 0.5 + 0.5 * math.sin(t * 9)
            cv2.circle(img, (x, y), int((22 + 7 * pulso) * s), config.COLOR_ACENTO, max(2, int(3 * s)), cv2.LINE_AA)
        texto(img, nombres_notas[dedo], x, y - 22 * s, tam=17 * s,
              color=(255, 255, 255) if estado.presionado[dedo] else color, peso=700, alinear="centro")


def dibujar_ondas(img, d: Diseno, ondas, t, duracion=0.45):
    """Anillos que se expanden desde la punta del dedo al tocar una nota."""
    for x, y, t0, color in ondas:
        p = (t - t0) / duracion
        if 0 <= p < 1:
            radio = int((16 + 46 * p) * d.s)
            grosor = max(1, int(5 * (1 - p) * d.s))
            cv2.circle(img, (int(x), int(y)), radio, mezclar_color(color, (255, 255, 255), p * 0.5),
                       grosor, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# Teclado
# ----------------------------------------------------------------------------
def geometria_teclado(rect, octava: int) -> dict:
    """Rectángulos de las teclas (dos octavas desde Do + Do final): midi -> (x1, y1, x2, y2, negra)."""
    x1, y1, x2, y2 = rect
    do = 12 * (octava + 1)
    blancas = [m for m in range(do, do + 25) if m % 12 in _TECLAS_BLANCAS]
    ancho = (x2 - x1) / len(blancas)
    teclas = {m: (int(x1 + i * ancho), y1, int(x1 + (i + 1) * ancho), y2, False)
              for i, m in enumerate(blancas)}
    alto_negra = int((y2 - y1) * 0.6)
    for m in range(do, do + 25):
        if m % 12 not in _TECLAS_BLANCAS:
            borde = teclas[m - 1][2]
            teclas[m] = (int(borde - ancho * 0.3), y1, int(borde + ancho * 0.3), y1 + alto_negra, True)
    return teclas


def dibujar_teclado(img, d: Diseno, octava, asignadas, brillo, objetivo, t):
    """
    asignadas: midi -> color del dedo que toca esa nota
    brillo:    midi -> (nivel 0..1, color) de las teclas que están sonando
    objetivo:  nota esperada en modo canción (o None)
    """
    s = d.s
    x1, y1, x2, y2 = d.teclado
    panel(img, (x1 - 8 * s, y1 - 8 * s), (x2 + 8 * s, y2 + 8 * s), alfa=0.6, radio=14 * s)
    teclas = geometria_teclado(d.teclado, octava)
    for dibujar_negras in (False, True):
        for midi, (a, b, c, e, negra) in teclas.items():
            if negra != dibujar_negras:
                continue
            base = (28, 28, 30) if negra else (238, 238, 236)
            relleno = base
            if midi in brillo:
                nivel, color = brillo[midi]
                relleno = mezclar_color(base, color, 0.35 + 0.65 * nivel)
            cv2.rectangle(img, (a, b), (c, e), relleno, -1)
            cv2.rectangle(img, (a, b), (c, e), (75, 75, 75), 1)
            if midi in asignadas:
                franja = max(4, int(7 * s))
                cv2.rectangle(img, (a + 3, e - franja - 2), (c - 3, e - 3), asignadas[midi], -1)
                texto(img, config.nombre_nota(midi), (a + c) / 2, e - franja - 9 * s,
                      tam=(12 if negra else 15) * s, color=(250, 250, 250) if negra else (45, 45, 45),
                      peso=700, alinear="centro", sombra=negra)
            if midi == objetivo:
                pulso = 0.5 + 0.5 * math.sin(t * 9)
                cv2.rectangle(img, (a + 2, b + 2), (c - 2, e - 2), config.COLOR_ACENTO, max(2, int((2 + 3 * pulso) * s)))
                texto(img, "▼", (a + c) / 2, y1 - (14 + 6 * pulso) * s, tam=24 * s,
                      color=config.COLOR_ACENTO, alinear="centro")


# ----------------------------------------------------------------------------
# Paneles
# ----------------------------------------------------------------------------
def dibujar_barra(img, d: Diseno, escala, rango, sensibilidad, fps, silenciado, segundos_grabando):
    """rango: texto de la octava ("Octava 4") o del rango de notas de una canción."""
    s = d.s
    panel(img, (0, 0), (d.w, d.alto_barra), alfa=0.7, radio=1)
    y = d.alto_barra * 0.68
    texto(img, "♪ Piano Virtual con Manos", 18 * s, y, tam=24 * s, color=config.COLOR_ACENTO, peso=700)
    elementos = [(f"Escala: {escala}", config.COLOR_TEXTO),
                 (rango, config.COLOR_TEXTO),
                 (f"Sensibilidad {sensibilidad:.0%}", config.COLOR_TEXTO),
                 (f"{fps:.0f} FPS", config.COLOR_TEXTO_SUAVE)]
    if silenciado:
        elementos.insert(0, ("Silenciado (M)", config.COLOR_ERROR))
    if segundos_grabando is not None:
        seg = int(segundos_grabando)
        elementos.insert(0, (f"● REC {seg // 60}:{seg % 60:02d}", (70, 70, 255)))
    x = d.w - 18 * s
    for txt, color in reversed(elementos):
        ancho = ancho_texto(txt, 16 * s, 500)
        texto(img, txt, x - ancho, y, tam=16 * s, color=color, peso=500)
        x -= ancho + 26 * s


def dibujar_panel_flexion(img, d: Diseno, estados, umbrales, notas):
    """Barras con la flexión de cada dedo sobre su reposo y la línea del umbral."""
    s = d.s
    ancho_barra, separacion, alto_max = 22 * s, 9 * s, 140 * s
    x0 = 18 * s
    y_base = d.teclado[1] - 66 * s   # deja libre la franja de la flecha ▼ del modo canción
    y_tope = y_base - alto_max - 40 * s
    ancho_panel = 10 * ancho_barra + 9 * separacion + 36 * s
    panel(img, (x0, y_tope), (x0 + ancho_panel, y_base + 28 * s), alfa=0.75)
    texto(img, "Flexión sobre el reposo (°)", x0 + 16 * s, y_tope + 24 * s, tam=14 * s,
          color=config.COLOR_ACENTO, peso=600)
    escala = alto_max / 90.0
    for p in range(10):
        lado, dedo = config.lado_y_dedo(p)
        estado = estados[lado]
        xa = int(x0 + 18 * s + p * (ancho_barra + separacion))
        xb = int(xa + ancho_barra)
        color = config.COLORES_DEDOS[dedo]
        cv2.rectangle(img, (xa, int(y_base - alto_max)), (xb, int(y_base)), (70, 70, 70), 1)
        if estado.frames > 0:
            valor = float(np.clip(estado.relativa[dedo], 0, 90))
            relleno = color if estado.presionado[dedo] else mezclar_color(color, (40, 40, 40), 0.45)
            cv2.rectangle(img, (xa + 1, int(y_base - valor * escala)), (xb - 1, int(y_base) - 1), relleno, -1)
        y_umbral = int(y_base - umbrales[dedo] * escala)
        cv2.line(img, (xa - 3, y_umbral), (xb + 3, y_umbral), (255, 255, 255), 1, cv2.LINE_AA)
        texto(img, config.nombre_nota(notas[p]), (xa + xb) / 2, y_base + 18 * s, tam=11 * s,
              color=config.COLOR_TEXTO_SUAVE, alinear="centro", sombra=False)


def dibujar_cancion(img, d: Diseno, cancion, lado_dedo_esperado, t):
    s = d.s
    ancho = min(d.w - 40 * s, 760 * s)
    x1 = (d.w - ancho) / 2
    y1 = d.alto_barra + 10 * s
    y2 = y1 + 96 * s
    panel(img, (x1, y1), (x1 + ancho, y2), alfa=0.78)
    centro = d.w / 2
    if cancion.terminada:
        texto(img, "¡Canción completada! ★", centro, y1 + 38 * s, tam=26 * s, color=config.COLOR_OK,
              peso=700, alinear="centro")
        duracion = cancion.t_fin - cancion.t_inicio
        texto(img, f"Precisión {cancion.precision:.0%}  ·  Tiempo {duracion:.1f} s  ·  "
                   f"Espacio: repetir  ·  C: siguiente", centro, y1 + 72 * s, tam=15 * s, alinear="centro")
        return
    texto(img, f"♪ {cancion.nombre}", x1 + 18 * s, y1 + 28 * s, tam=18 * s, color=config.COLOR_ACENTO, peso=700)
    texto(img, f"{cancion.posicion}/{len(cancion.notas)}  ·  Precisión {cancion.precision:.0%}",
          x1 + ancho - 18 * s, y1 + 28 * s, tam=14 * s, color=config.COLOR_TEXTO_SUAVE, alinear="der")
    x = x1 + 18 * s
    for i, midi in enumerate(cancion.proximas(10)):
        nombre = config.nombre_nota(midi)
        tam, peso = (26 * s, 700) if i == 0 else (18 * s, 500)
        color = config.COLOR_ACENTO if i == 0 else mezclar_color(config.COLOR_TEXTO, config.COLOR_PANEL, 0.12 * i)
        texto(img, nombre, x, y1 + 64 * s, tam=tam, color=color, peso=peso)
        x += ancho_texto(nombre, tam, peso) + 18 * s
        if x > x1 + ancho - 60 * s:
            break
    if lado_dedo_esperado is not None:
        lado, dedo = lado_dedo_esperado
        mano = "izquierda" if lado == "izq" else "derecha"
        texto(img, f"Usa el {config.NOMBRES_DEDOS[dedo].lower()} de la mano {mano}", x1 + 18 * s, y1 + 86 * s,
              tam=13 * s, color=config.COLORES_DEDOS[dedo], peso=600)
    progreso = cancion.posicion / len(cancion.notas)
    cv2.rectangle(img, (int(x1 + 10 * s), int(y2 - 4 * s)), (int(x1 + 10 * s + (ancho - 20 * s) * progreso), int(y2 - 2 * s)),
                  config.COLOR_OK, -1)


_AYUDA = (
    ("Levanta las manos frente a la cámara, con los dedos relajados.", None),
    ("Dobla un dedo con rapidez, como si presionaras una tecla.", None),
    ("Cada dedo es una nota: mano izquierda graves, derecha agudas.", None),
    ("Golpe más rápido = nota más fuerte. El anillo del dedo muestra cuánto falta.", None),
    ("E", "Cambiar escala (mayor, pentatónica, menor, blues)"),
    ("↑ ↓  o  + −", "Subir / bajar una octava"),
    ("[  ]", "Bajar / subir la sensibilidad"),
    ("C", "Modo canción (pasa a la siguiente)"),
    ("Espacio", "Reiniciar la canción"),
    ("R", "Grabar / guardar en WAV"),
    ("D", "Panel de flexión de los dedos"),
    ("M", "Silenciar"),
    ("F", "Pantalla completa"),
    ("H", "Mostrar / ocultar esta ayuda"),
    ("Q / Esc", "Salir"),
)


def dibujar_ayuda(img, d: Diseno):
    s = d.s
    ancho, alto = 700 * s, 470 * s
    x1, y1 = (d.w - ancho) / 2, max(d.alto_barra + 10 * s, (d.h - alto) / 2 - 40 * s)
    panel(img, (x1, y1), (x1 + ancho, y1 + alto), alfa=0.88)
    texto(img, "Cómo tocar", x1 + 28 * s, y1 + 40 * s, tam=22 * s, color=config.COLOR_ACENTO, peso=700)
    y = y1 + 72 * s
    titulo_teclas = False
    for clave, descripcion in _AYUDA:
        if descripcion is None:
            texto(img, "• " + clave, x1 + 28 * s, y, tam=15 * s)
        else:
            if not titulo_teclas:
                y += 12 * s
                texto(img, "Teclas", x1 + 28 * s, y, tam=20 * s, color=config.COLOR_ACENTO, peso=700)
                y += 30 * s
                titulo_teclas = True
            texto(img, clave, x1 + 40 * s, y, tam=15 * s, color=config.COLOR_ACENTO, peso=700)
            texto(img, descripcion, x1 + 190 * s, y, tam=15 * s)
        y += 24 * s


def dibujar_sin_manos(img, d: Diseno, imagen_oscura=False):
    if imagen_oscura:
        texto(img, "La imagen de la cámara está negra", d.w / 2, d.h * 0.42, tam=32 * d.s, peso=700,
              color=config.COLOR_ERROR, alinear="centro")
        texto(img, "¿Está cerrada la tapa de privacidad de la cámara o falta luz?", d.w / 2,
              d.h * 0.42 + 38 * d.s, tam=17 * d.s, color=config.COLOR_TEXTO_SUAVE, alinear="centro")
        return
    texto(img, "Muestra tus manos a la cámara", d.w / 2, d.h * 0.42, tam=32 * d.s, peso=700, alinear="centro")
    texto(img, "Dedos relajados  ·  dobla un dedo para tocar  ·  H = ayuda", d.w / 2, d.h * 0.42 + 38 * d.s,
          tam=17 * d.s, color=config.COLOR_TEXTO_SUAVE, alinear="centro")


def dibujar_mensaje(img, d: Diseno, mensaje: str, restante: float):
    """Aviso temporal sobre el teclado; se desvanece en el último 0.4 s."""
    if restante <= 0 or not mensaje:
        return
    s = d.s
    alfa = min(1.0, restante / 0.4)
    tam = 17 * s
    ancho = ancho_texto(mensaje, tam, 600) + 44 * s
    y = d.teclado[1] - 34 * s
    panel(img, (d.w / 2 - ancho / 2, y - 30 * s), (d.w / 2 + ancho / 2, y + 12 * s), alfa=0.85 * alfa)
    texto(img, mensaje, d.w / 2, y, tam=tam, color=mezclar_color(config.COLOR_PANEL, config.COLOR_TEXTO, alfa),
          peso=600, alinear="centro", sombra=False)
