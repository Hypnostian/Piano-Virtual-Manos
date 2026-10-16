"""
Pruebas automáticas del piano (no necesitan cámara).

    python test_piano.py        # o:  python -m pytest test_piano.py
"""

import tempfile
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import camara
import config
from canciones import CANCIONES, ModoCancion, calcular_tramos, notas
from manos import EstadoMano, ManoDetectada, asignar_manos, flexion_dedos
from sonido import Grabadora, Sintetizador, sintetizar_nota

FPS = 30.0
UMBRALES = np.array(config.UMBRAL_PRESION)


def _rotacion_aleatoria(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _simular(curvas_indice, ruido=0.0, semilla=0):
    """Pasa una serie de flexiones del índice (los demás dedos quietos) por EstadoMano."""
    rng = np.random.default_rng(semilla)
    estado, eventos = EstadoMano(), []
    for i, valor in enumerate(curvas_indice):
        flexion = np.array([50.0, valor, 60.0, 60.0, 55.0]) + rng.normal(0, ruido, 5)
        for dedo, _ in estado.actualizar(flexion, i / FPS, UMBRALES):
            eventos.append((i / FPS, dedo))
    return eventos


def _rampa(desde, hasta, segundos):
    return list(np.linspace(desde, hasta, max(2, int(segundos * FPS))))


# ---------------------------------------------------------------- geometría
def test_flexion_dedo_recto_es_cero_y_doblado_suma_angulos():
    # Muñeca en el origen, nudillos a 8 cm y dedos rectos hacia +y
    mundo = np.zeros((21, 3))
    for cadena, x in zip(config.CADENAS_DEDOS, (-0.03, -0.015, 0.0, 0.015, 0.03)):
        for j, idx in enumerate(cadena[1:]):
            mundo[idx] = [x, 0.08 + 0.02 * j, 0.0]
    # Índice doblado 30° en cada una de sus 3 articulaciones (plano y-z)
    punto = mundo[5].copy()
    for j, idx in enumerate((6, 7, 8), start=1):
        angulo = np.radians(30 * j)
        punto = punto + 0.02 * np.array([0.0, np.cos(angulo), np.sin(angulo)])
        mundo[idx] = punto
    flex = flexion_dedos(mundo.astype(np.float32))
    assert flex[2] < 1, flex             # medio: recto
    assert abs(flex[1] - 90) < 5, flex   # índice: 3 x 30° (+ la leve inclinación del nudillo)


def test_mover_o_girar_la_mano_no_cambia_la_flexion():
    """La gran diferencia con detectar la 'velocidad de la punta': mover la mano no toca notas."""
    rng = np.random.default_rng(1)
    mundo = rng.normal(0, 0.03, size=(21, 3)).astype(np.float32)
    original = flexion_dedos(mundo)
    for _ in range(20):
        R = _rotacion_aleatoria(rng)
        movida = (mundo @ R.T) * rng.uniform(0.7, 1.4) + rng.normal(0, 0.2, 3)
        assert np.allclose(flexion_dedos(movida.astype(np.float32)), original, atol=0.05)


# ---------------------------------------------------------------- detector de golpes
def test_golpe_rapido_toca_una_sola_nota():
    serie = [60.0] * 30 + _rampa(60, 105, 0.15) + [105.0] * 6 + _rampa(105, 60, 0.2) + [60.0] * 30
    eventos = _simular(serie, ruido=1.5)
    assert [d for _, d in eventos] == [1], eventos


def test_doblar_el_dedo_lentamente_no_toca():
    serie = [60.0] * 30 + _rampa(60, 105, 3.0) + [105.0] * 30
    assert _simular(serie, ruido=1.0) == []


def test_temblor_y_ruido_no_tocan():
    assert _simular([65.0] * int(8 * FPS), ruido=4.0, semilla=3) == []


def test_repetir_nota_sin_estirar_todo_el_dedo():
    serie = ([60.0] * 30 + _rampa(60, 105, 0.15) + [105.0] * 4 + _rampa(105, 82, 0.15)
             + _rampa(82, 125, 0.15) + [125.0] * 4 + _rampa(125, 60, 0.3) + [60.0] * 10)
    assert [d for _, d in _simular(serie)] == [1, 1]


def test_asignar_manos_por_posicion_y_continuidad():
    def mano(x, etiqueta):
        puntos = np.zeros((21, 2), np.float32)
        puntos[0] = [x, 0.5]
        return ManoDetectada(puntos, np.zeros((21, 3), np.float32), etiqueta, 0.9)

    # Con dos manos manda la posición, aunque las etiquetas vengan mal
    dos = asignar_manos([mano(0.8, "Left"), mano(0.2, "Left")], {})
    assert dos["izq"].puntos[0, 0] == np.float32(0.2) and dos["der"].puntos[0, 0] == np.float32(0.8)
    # Con una mano se respeta la continuidad aunque la etiqueta cambie
    previa = {"der": np.array([0.7, 0.5])}
    assert list(asignar_manos([mano(0.72, "Left")], previa)) == ["der"]
    assert list(asignar_manos([mano(0.3, "Left")], {})) == ["izq"]


# ---------------------------------------------------------------- cámara
_RUTA_INTEGRADA = r"\\?\usb#vid_04f2&pid_b7b9&mi_00#6&261941b5&1&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\global"
_RUTA_EXTERNA = r"\\?\usb#vid_046d&pid_085c&mi_00#7&1a2b3c&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\global"
_RUTA_VIRTUAL = r"\\?\root#image#0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\global"
_PLD_TAPA_FRONTAL = bytes.fromhex("82000000000000002401000000000000FFFFFFFF")  # leído del portátil real


def test_ruta_de_la_camara_a_id_de_windows():
    msmf = _RUTA_INTEGRADA.replace("65e8773d-8f56-11d0-a3b9-00a0c9223196", "e5323777-f976-4f5b-9b55-b94699c46e44")
    esperado = r"USB\VID_04F2&PID_B7B9&MI_00\6&261941B5&1&0000"
    assert camara.instancia_desde_ruta(_RUTA_INTEGRADA) == esperado == camara.instancia_desde_ruta(msmf)
    assert camara.instancia_desde_ruta(_RUTA_VIRTUAL) == r"ROOT\IMAGE\0000"
    assert camara.instancia_desde_ruta("") == ""
    assert camara.describir_ubicacion(_PLD_TAPA_FRONTAL) == (camara.PANEL_FRONTAL, True)


def test_solo_se_usa_la_camara_integrada():
    """Como en el portátil real: una cámara virtual ocupa el índice 0, pero se elige la integrada."""
    def info(indice, nombre, ruta):
        return SimpleNamespace(index=indice, backend=700, name=nombre, path=ruta)

    integrada = camara.instancia_desde_ruta(_RUTA_INTEGRADA)
    propiedades = {
        (integrada, camara.UBICACION_FISICA): _PLD_TAPA_FRONTAL,
        (integrada, camara.EN_CONTENEDOR_DEL_EQUIPO): b"\xff",
        (r"ROOT\IMAGE\0000", camara.EN_CONTENEDOR_DEL_EQUIPO): b"\xff",   # la virtual también "es del equipo"
        (camara.instancia_desde_ruta(_RUTA_EXTERNA), camara.EN_CONTENEDOR_DEL_EQUIPO): b"\x00",
    }
    def leer(instancia, clave):
        return propiedades.get((instancia, clave))

    lista = [info(0, "Sharing Camera", _RUTA_VIRTUAL), info(1, "Logitech C920", _RUTA_EXTERNA),
             info(2, "Integrated Camera", _RUTA_INTEGRADA), info(3, "OBS Virtual Camera", "")]
    camaras = [camara.clasificar(c, leer) for c in lista]
    assert [c.integrada for c in camaras] == [False, False, True, False]
    elegida = camara.elegir_integrada(camaras)
    assert elegida.indice == 2 and "panel frontal" in elegida.detalle
    # Si la integrada no está disponible, NO se elige ninguna otra cámara
    assert camara.elegir_integrada([c for c in camaras if c.indice != 2]) is None


# ---------------------------------------------------------------- música y sonido
def test_notas_de_los_dedos():
    assert config.notas_para_dedos(config.ESCALAS[0][1], 4) == [60, 62, 64, 65, 67, 69, 71, 72, 74, 76]
    assert config.nombre_nota(60) == "Do4" and config.nombre_nota(70) == "La#4"
    for p in range(10):
        assert config.posicion_dedo(*config.lado_y_dedo(p)) == p


def test_sintesis_de_todas_las_notas():
    for midi in range(36, 97, 5):
        onda = sintetizar_nota(midi)
        assert onda.dtype == np.float32 and onda.shape[1] == 2
        assert np.isfinite(onda).all() and 0.5 < np.abs(onda).max() <= 0.901


def test_grabacion_genera_wav():
    sinte = Sintetizador(activo=False)
    grabadora = Grabadora()
    grabadora.iniciar(10.0)
    for i, midi in enumerate([60, 64, 67, 72, 72]):
        grabadora.registrar(10.0 + 0.3 * i, midi, 0.8)
    with tempfile.TemporaryDirectory() as carpeta:
        ruta = grabadora.detener(sinte, Path(carpeta))
        with wave.open(str(ruta)) as w:
            assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (2, 2, 44100)
            assert 2.0 < w.getnframes() / 44100 < 6.0


def test_modo_cancion():
    normales = config.notas_para_dedos(config.ESCALAS[0][1], 4)
    for indice, (_, notas) in enumerate(CANCIONES):
        cancion = ModoCancion(indice)
        for i, midi in enumerate(notas):
            dedos = cancion.asignacion_actual() or normales
            assert len(dedos) == 10 and dedos == sorted(dedos)
            assert midi in dedos                                  # cada nota cae en un dedo
            assert cancion.registrar([midi], float(i)) is True
        assert cancion.terminada and cancion.precision == 1.0
    cancion = ModoCancion(1)
    assert cancion.registrar([60], 0.0) is False                   # nota equivocada
    assert cancion.registrar([62, cancion.esperada], 0.1) is True   # correcta + vecina
    assert cancion.aciertos == 1 and cancion.errores == 1


def test_leer_notas_en_varios_formatos():
    assert notas("Do4 Re#4 Mib4 Sol4") == notas("C4 D#4 Eb4 G4") == [60, 63, 63, 67]
    assert notas("1 3 5 1° 3°") == [60, 64, 67, 72, 76]           # kalimba
    assert notas("1 (135) 2") == [60, 67, 62]                       # acorde -> nota más aguda
    melodia = notas("Mi4 Mi4 | Fa4 Sol4\nLa4")
    assert melodia == [64, 64, 65, 67, 69] and melodia.frases == [0, 2, 4]
    assert notas("   ") == []


def test_las_notas_solo_cambian_si_la_cancion_lo_necesita():
    nombres = [nombre for nombre, _ in CANCIONES]
    for nombre in ("Himno de la Alegría (Beethoven)", "Estrellita, ¿dónde estás?", "Jingle Bells"):
        assert not ModoCancion(nombres.index(nombre)).necesita_otras_notas
    elisa = ModoCancion(nombres.index("Para Elisa (Beethoven)"))
    assert elisa.necesita_otras_notas and len(set(elisa.notas)) == 12
    assert [tramo.inicio for tramo in elisa.tramos] == [0, 35]      # cambia al empezar una frase
    assert 75 in elisa.tramos[0].asignacion and 68 in elisa.tramos[0].asignacion   # Re#5 y Sol#4
    # Una sola frase con 12 notas distintas se reparte en tramos de máximo 10 notas
    tramos = calcular_tramos(list(range(60, 72)))
    assert [t.fin - t.inicio for t in tramos] == [10, 2]
    assert all(set(range(60, 72)[t.inicio:t.fin]) <= set(t.asignacion) for t in tramos)


if __name__ == "__main__":
    pruebas = [(nombre, f) for nombre, f in globals().items() if nombre.startswith("test_")]
    fallidas = 0
    for nombre, prueba in pruebas:
        try:
            prueba()
            print(f"  OK    {nombre}")
        except AssertionError as error:
            fallidas += 1
            print(f"  FALLA {nombre}: {error}")
    print(f"\n{len(pruebas) - fallidas}/{len(pruebas)} pruebas superadas")
    raise SystemExit(1 if fallidas else 0)
