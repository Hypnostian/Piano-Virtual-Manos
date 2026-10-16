"""
Piano Virtual con Manos
=======================
Toca un piano doblando los dedos frente a la cámara, sin tocar nada.
Siempre usa la cámara integrada del portátil (nunca cámaras virtuales, del celular ni USB).

    python app.py                   # abrir el piano con la cámara integrada
    python app.py --listar-camaras  # ver qué cámaras hay y cuál se usa
    python app.py --video demo.mp4  # usar un video en vez de la cámara
    python app.py --panel           # iniciar mostrando la flexión de cada dedo
"""

import argparse
import os
import time

os.environ.setdefault("GLOG_minloglevel", "2")   # menos mensajes internos de MediaPipe
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")   # Media Foundation abre más rápido

import cv2
import numpy as np

import camara
import config
import interfaz as ui
from canciones import CANCIONES, ModoCancion
from manos import DetectorManos, EstadoMano, asignar_manos, flexion_dedos
from sonido import Grabadora, Sintetizador

FLECHAS_ARRIBA = {2490368, 65362, 63232}   # códigos de flecha arriba (Windows, Linux, macOS)
FLECHAS_ABAJO = {2621440, 65364, 63233}


def intensidad_por_velocidad(velocidad: float) -> float:
    """Igual que un teclado MIDI: cuanto más rápido baja el dedo, más fuerte suena."""
    x = (velocidad - config.VEL_SUAVE) / (config.VEL_FUERTE - config.VEL_SUAVE)
    return float(np.clip(config.INTENSIDAD_MIN + (1 - config.INTENSIDAD_MIN) * x, config.INTENSIDAD_MIN, 1.0))


class PianoVirtual:
    """Estado de la aplicación: procesa cada fotograma, toca las notas y dibuja la interfaz."""

    def __init__(self, sintetizador: Sintetizador, mostrar_panel=False):
        self.sinte = sintetizador
        self.grabadora = Grabadora()
        self.estados = {"izq": EstadoMano(), "der": EstadoMano()}
        self.manos = {}
        self.munecas = {}
        self.indice_escala = 0
        self.octava = config.OCTAVA_INICIAL
        self.sensibilidad = 1.0
        self.cancion = None
        self.mostrar_panel = mostrar_panel
        self.mostrar_ayuda = False
        self.mensaje = ("", 0.0)
        self.luces = {}       # midi -> (t, color): última vez que la tecla sonó o estuvo presionada
        self.destellos = {}   # midi -> (t, color): acierto / error en modo canción
        self.ondas = []       # (x, y, t, color)
        self.fps = 0.0
        self.notas_tocadas = 0
        self.imagen_oscura = False
        self._actualizar_notas()
        self.sinte.precalentar()

    # ---------------------------------------------------------------- música
    def _actualizar_notas(self):
        self.notas = config.notas_para_dedos(config.ESCALAS[self.indice_escala][1], self.octava)
        self.sinte.precargar(self.notas)

    def _aplicar_tramo(self, t: float, avisar=True):
        """
        En modo canción, si la canción necesita otras notas, los dedos toman las notas del
        tramo actual. Si la canción cabe en las notas normales, no se cambia nada.
        """
        asignacion = self.cancion.asignacion_actual() if self.cancion else None
        if asignacion is None or asignacion == self.notas:
            return
        self.notas = list(asignacion)
        self.sinte.precargar(self.notas)
        if avisar:
            numero = self.cancion.tramo_actual() + 1
            self.avisar(f"♪ Cambian las notas de los dedos (parte {numero} de {len(self.cancion.tramos)})", t, 3.0)

    def umbrales(self) -> np.ndarray:
        return np.array(config.UMBRAL_PRESION) / self.sensibilidad

    def avisar(self, texto: str, t: float, duracion=2.2):
        self.mensaje = (texto, t + duracion)

    # ---------------------------------------------------------------- procesamiento
    def procesar(self, frame, t: float, detector: DetectorManos) -> list:
        """Detecta las manos, actualiza cada dedo y toca las notas. Devuelve las notas tocadas."""
        self.imagen_oscura = float(frame[::16, ::16].mean()) < 12   # ¿tapa de privacidad cerrada?
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.manos = asignar_manos(detector.detectar(rgb, t * 1000), self.munecas)
        self.munecas = {lado: mano.puntos[0] for lado, mano in self.manos.items()}
        h, w = frame.shape[:2]
        umbrales = self.umbrales()
        tocadas = []
        for lado, estado in self.estados.items():
            mano = self.manos.get(lado)
            if mano is None:
                estado.perder(t)
                continue
            for dedo, velocidad in estado.actualizar(flexion_dedos(mano.mundo), t, umbrales):
                midi = self.notas[config.posicion_dedo(lado, dedo)]
                intensidad = intensidad_por_velocidad(velocidad)
                color = config.COLORES_DEDOS[dedo]
                self.sinte.tocar(midi, intensidad)
                self.grabadora.registrar(t, midi, intensidad)
                x, y = mano.puntos[config.PUNTAS[dedo]] * (w, h)
                self.ondas.append((x, y, t, color))
                tocadas.append(midi)
            for dedo in np.flatnonzero(estado.presionado):   # teclas sostenidas
                self.luces[self.notas[config.posicion_dedo(lado, dedo)]] = (t, config.COLORES_DEDOS[dedo])

        self.notas_tocadas += len(tocadas)
        if self.cancion is not None:
            acierto = self.cancion.registrar(tocadas, t)
            if acierto is not None:
                midi = self.cancion.notas[self.cancion.posicion - 1] if acierto else tocadas[0]
                self.destellos[midi] = (t, config.COLOR_OK if acierto else config.COLOR_ERROR)
            if acierto:
                self._aplicar_tramo(t)
        return tocadas

    # ---------------------------------------------------------------- dibujo
    def dibujar(self, img, t: float):
        d = ui.Diseno(img.shape[1], img.shape[0])
        umbrales = self.umbrales()
        esperada = self.cancion.esperada if self.cancion else None
        dedo_de_nota = {self.notas[p]: config.lado_y_dedo(p) for p in range(10)}
        objetivo = dedo_de_nota.get(esperada)

        for lado, mano in self.manos.items():
            nombres = [config.nombre_nota(self.notas[config.posicion_dedo(lado, dedo)]) for dedo in range(5)]
            dedo_objetivo = objetivo[1] if objetivo and objetivo[0] == lado else None
            ui.dibujar_mano(img, d, mano.puntos, self.estados[lado], umbrales, nombres, dedo_objetivo, t)
        self.ondas = [o for o in self.ondas if t - o[2] < 0.45]
        ui.dibujar_ondas(img, d, self.ondas, t)

        asignadas = {self.notas[p]: config.COLORES_DEDOS[config.lado_y_dedo(p)[1]] for p in range(10)}
        brillo = {}
        for registro, duracion in ((self.luces, 0.35), (self.destellos, 0.45)):
            for midi, (t0, color) in list(registro.items()):
                nivel = 1.0 - (t - t0) / duracion
                if nivel > 0:
                    brillo[midi] = (nivel, color)
                else:
                    del registro[midi]
        octava_teclado = min(self.notas) // 12 - 1   # el teclado empieza en el Do de la nota más grave
        ui.dibujar_teclado(img, d, octava_teclado, asignadas, brillo, esperada, t)

        segundos = self.grabadora.duracion(t) if self.grabadora.grabando else None
        if self.cancion is not None and self.cancion.asignacion_actual() is not None:
            escala = "notas de la canción"
            rango = f"{config.nombre_nota(min(self.notas))} a {config.nombre_nota(max(self.notas))}"
        else:
            escala, rango = config.ESCALAS[self.indice_escala][0], f"Octava {self.octava}"
        ui.dibujar_barra(img, d, escala, rango, self.sensibilidad, self.fps, self.sinte.silenciado, segundos)
        if self.cancion is not None:
            ui.dibujar_cancion(img, d, self.cancion, objetivo, t)
        if self.mostrar_panel:
            ui.dibujar_panel_flexion(img, d, self.estados, umbrales, self.notas)
        if not self.manos and not self.mostrar_ayuda:
            ui.dibujar_sin_manos(img, d, self.imagen_oscura)
        if self.mostrar_ayuda:
            ui.dibujar_ayuda(img, d)
        else:
            ui.texto(img, "H = ayuda", d.w - 18 * d.s, d.teclado[1] - 16 * d.s, tam=13 * d.s,
                     color=config.COLOR_TEXTO_SUAVE, alinear="der")
        ui.dibujar_mensaje(img, d, self.mensaje[0], self.mensaje[1] - t)

    # ---------------------------------------------------------------- teclado de la PC
    def manejar_tecla(self, tecla: int, t: float) -> bool:
        """Aplica la acción de una tecla. Devuelve False si hay que salir."""
        if tecla == -1:
            return True
        if tecla in FLECHAS_ARRIBA:
            self._cambiar_octava(+1, t)
            return True
        if tecla in FLECHAS_ABAJO:
            self._cambiar_octava(-1, t)
            return True
        codigo = tecla & 0xFF
        if codigo == 27:
            return False
        c = chr(codigo).lower()
        acciones = {
            "q": None,
            "+": lambda: self._cambiar_octava(+1, t), "=": lambda: self._cambiar_octava(+1, t),
            "-": lambda: self._cambiar_octava(-1, t), "_": lambda: self._cambiar_octava(-1, t),
            "e": lambda: self._cambiar_escala(t),
            "]": lambda: self._cambiar_sensibilidad(+0.1, t),
            "[": lambda: self._cambiar_sensibilidad(-0.1, t),
            "c": lambda: self._siguiente_cancion(t),
            " ": lambda: self._reiniciar_cancion(t),
            "r": lambda: self.alternar_grabacion(t),
            "d": self._alternar_panel,
            "m": lambda: self._silenciar(t),
            "h": self._alternar_ayuda,
        }
        if c not in acciones:
            return True
        if acciones[c] is None:   # q = salir
            return False
        acciones[c]()
        return True

    def _cambiar_octava(self, paso, t):
        if self.cancion is not None:
            self.avisar("Sal del modo canción (C) para cambiar la octava", t)
            return
        self.octava = int(np.clip(self.octava + paso, config.OCTAVA_MIN, config.OCTAVA_MAX))
        self._actualizar_notas()
        self.avisar(f"Octava {self.octava}", t)

    def _cambiar_escala(self, t):
        if self.cancion is not None:
            self.avisar("Sal del modo canción (C) para cambiar la escala", t)
            return
        self.indice_escala = (self.indice_escala + 1) % len(config.ESCALAS)
        self._actualizar_notas()
        self.avisar(f"Escala: {config.ESCALAS[self.indice_escala][0]}", t)

    def _cambiar_sensibilidad(self, paso, t):
        self.sensibilidad = round(float(np.clip(self.sensibilidad + paso, config.SENSIBILIDAD_MIN,
                                                config.SENSIBILIDAD_MAX)), 1)
        self.avisar(f"Sensibilidad {self.sensibilidad:.0%}", t)

    def _siguiente_cancion(self, t):
        siguiente = 0 if self.cancion is None else self.cancion.indice_cancion + 1
        if siguiente >= len(CANCIONES):
            self.cancion = None
            self._actualizar_notas()   # vuelven las notas normales de la escala
            self.avisar("Modo libre", t)
            return
        self.cancion = ModoCancion(siguiente)
        self.indice_escala, self.octava = 0, 4   # las canciones parten de Do mayor, octava 4
        self._actualizar_notas()
        self._aplicar_tramo(t, avisar=False)      # solo cambia si la canción lo necesita
        extra = " · usa otras notas" if self.cancion.necesita_otras_notas else ""
        self.avisar(f"Modo canción: {self.cancion.nombre}{extra}", t, 3.0)

    def _reiniciar_cancion(self, t):
        if self.cancion is not None:
            self.cancion.reiniciar()
            self._aplicar_tramo(t, avisar=False)
            self.avisar("Canción reiniciada", t)

    def alternar_grabacion(self, t):
        if self.grabadora.grabando:
            ruta = self.grabadora.detener(self.sinte)
            if ruta is None:
                self.avisar("Grabación vacía: no se tocó ninguna nota", t)
            else:
                self.avisar(f"Guardado: grabaciones/{ruta.name}", t, 4.0)
                print(f"Grabación guardada en {ruta}")
        else:
            self.grabadora.iniciar(t)
            self.avisar("● Grabando... pulsa R otra vez para guardar", t)

    def _alternar_panel(self):
        self.mostrar_panel = not self.mostrar_panel

    def _alternar_ayuda(self):
        self.mostrar_ayuda = not self.mostrar_ayuda

    def _silenciar(self, t):
        self.sinte.silenciado = not self.sinte.silenciado
        self.avisar("Sonido desactivado" if self.sinte.silenciado else "Sonido activado", t)


# ============================================================================
# Cámara / video
# ============================================================================
def leer_argumentos():
    p = argparse.ArgumentParser(description="Piano virtual controlado con los dedos (MediaPipe + OpenCV).")
    p.add_argument("--listar-camaras", action="store_true",
                   help="mostrar las cámaras detectadas y cuál es la integrada (la única que se usa)")
    p.add_argument("--video", help="usar un archivo de video en lugar de la cámara")
    p.add_argument("--sin-espejo", action="store_true", help="no voltear la imagen horizontalmente")
    p.add_argument("--sin-sonido", action="store_true", help="no reproducir audio")
    p.add_argument("--panel", action="store_true", help="mostrar desde el inicio el panel de flexión")
    p.add_argument("--sin-ventana", action="store_true", help="procesar sin mostrar ventana (pruebas)")
    p.add_argument("--guardar-video", help="guardar la salida con la interfaz en un archivo .mp4")
    p.add_argument("--max-fotogramas", type=int, default=0, help="detenerse tras N fotogramas (pruebas)")
    return p.parse_args()


def abrir_fuente(args):
    """
    Abre la cámara integrada (o el video). Devuelve (captura, fps del video o None si es
    cámara, descripción de la fuente). Nunca abre otra cámara que no sea la integrada.
    """
    if args.video:
        captura = cv2.VideoCapture(args.video)
        if not captura.isOpened():
            raise SystemExit(f"No se pudo abrir el video: {args.video}")
        return captura, captura.get(cv2.CAP_PROP_FPS) or 30.0, f"Video: {os.path.basename(args.video)}"
    try:
        captura, cam = camara.abrir_camara_integrada(config.ANCHO, config.ALTO)
    except camara.CamaraNoDisponible as error:
        raise SystemExit(f"\n{error}\n")
    return captura, None, f"Cámara integrada: {cam.nombre} · {cam.detalle}"


def preparar_fotograma(frame, espejo: bool, es_video: bool):
    """Efecto espejo (como una selfie) y tamaño uniforme de 720 px de alto."""
    if espejo:
        frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    if es_video and w / h < 1.3:   # video vertical (celular): se centra en un lienzo 16:9
        nuevo_ancho = int(h * 16 / 9)
        izquierda = (nuevo_ancho - w) // 2
        frame = cv2.copyMakeBorder(frame, 0, 0, izquierda, nuevo_ancho - w - izquierda,
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
        h, w = frame.shape[:2]
    if h != config.ALTO:
        frame = cv2.resize(frame, (round(w * config.ALTO / h), config.ALTO), interpolation=cv2.INTER_LINEAR)
    return frame


def main():
    args = leer_argumentos()
    if args.listar_camaras:
        camara.imprimir_camaras()
        return
    print("=" * 62)
    print("  ♪ Piano Virtual con Manos  ·  MediaPipe + OpenCV + pygame")
    print("=" * 62)
    print("  Dobla un dedo frente a la cámara para tocar su nota.")
    print("  H ayuda · E escala · C canción · R grabar · D panel · Q salir\n")

    captura, fps_video, fuente = abrir_fuente(args)
    print(f"  {fuente}\n")
    es_video = fps_video is not None
    detector = DetectorManos()
    sintetizador = Sintetizador(activo=not args.sin_sonido)
    piano = PianoVirtual(sintetizador, mostrar_panel=args.panel)
    piano.avisar(fuente.split(" · ")[0], 0.0, 4.0)

    con_ventana = not args.sin_ventana
    if con_ventana:
        cv2.namedWindow(config.NOMBRE_VENTANA, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(config.NOMBRE_VENTANA, config.ANCHO, config.ALTO)
    escritor = None
    pantalla_completa = False
    fallos = 0
    n = 0
    t = 0.0
    inicio = time.perf_counter()
    anterior = inicio
    try:
        while True:
            ok, frame = captura.read()
            if not ok:
                if es_video:
                    break
                fallos += 1
                if fallos > 60:
                    print("La cámara integrada dejó de enviar imágenes. ¿La está usando otra app "
                          "(Teams, Zoom, Meet...)? Ciérrala y vuelve a abrir el piano.")
                    break
                continue
            fallos = 0
            frame = preparar_fotograma(frame, espejo=not args.sin_espejo, es_video=es_video)
            ahora = time.perf_counter()
            t = n / fps_video if es_video else ahora - inicio
            if n:
                instantaneo = 1.0 / max(ahora - anterior, 1e-3)
                piano.fps = instantaneo if piano.fps == 0 else 0.9 * piano.fps + 0.1 * instantaneo
            anterior = ahora

            piano.procesar(frame, t, detector)
            piano.dibujar(frame, t)

            if args.guardar_video:
                if escritor is None:
                    escritor = cv2.VideoWriter(args.guardar_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                               fps_video or 30.0, (frame.shape[1], frame.shape[0]))
                escritor.write(frame)
            n += 1
            if args.max_fotogramas and n >= args.max_fotogramas:
                break
            if not con_ventana:
                continue

            cv2.imshow(config.NOMBRE_VENTANA, frame)
            espera = 1
            if es_video:   # reproducir el video a su velocidad real
                espera = max(1, int((n / fps_video - (time.perf_counter() - inicio)) * 1000))
            tecla = cv2.waitKeyEx(espera)
            if tecla != -1 and chr(tecla & 0xFF).lower() == "f":
                pantalla_completa = not pantalla_completa
                cv2.setWindowProperty(config.NOMBRE_VENTANA, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if pantalla_completa else cv2.WINDOW_NORMAL)
            elif not piano.manejar_tecla(tecla, t):
                break
            if cv2.getWindowProperty(config.NOMBRE_VENTANA, cv2.WND_PROP_VISIBLE) < 1:
                break   # se cerró la ventana con la X
    finally:
        if piano.grabadora.grabando:
            piano.alternar_grabacion(t)
        captura.release()
        detector.cerrar()
        sintetizador.cerrar()
        if escritor is not None:
            escritor.release()
        cv2.destroyAllWindows()
        segundos = time.perf_counter() - inicio
        print(f"Fin. Fotogramas: {n} · notas tocadas: {piano.notas_tocadas} · "
              f"{n / max(segundos, 1e-3):.1f} FPS promedio")


if __name__ == "__main__":
    main()
