"""
Selección de cámara: SIEMPRE la cámara integrada del equipo, nunca otra.

Por qué no basta con cv2.VideoCapture(0): Windows numera las cámaras de forma distinta
en cada backend y en esa lista aparecen cámaras virtuales ("Sharing Camera",
"OBS Virtual Camera"...), la cámara del celular o webcams USB. En el portátil donde se
desarrolló este proyecto, el índice 0 de DirectShow es una cámara virtual.

Cómo se reconoce la cámara integrada (Windows), sin encender ninguna cámara:
  1. Se listan las cámaras con su ruta de hardware (librería cv2-enumerate-cameras).
  2. Se descartan las virtuales: no tienen ruta o están en un bus de software (ROOT, SWD).
  3. Se consulta a Windows la "ubicación física" del dispositivo (dato ACPI _PLD del
     fabricante). Solo las cámaras de fábrica la tienen, por ejemplo "en la tapa, panel
     frontal". Las webcams USB externas no la tienen.
"""

import os
import uuid
from dataclasses import dataclass

import cv2

try:
    from cv2_enumerate_cameras import enumerate_cameras
except ImportError:
    enumerate_cameras = None

BUSES_VIRTUALES = ("ROOT", "SWD", "HTREE")   # dispositivos creados por software
PANELES = ("superior", "inferior", "izquierdo", "derecho", "frontal", "trasero", "desconocido")
PANEL_FRONTAL = 4
PALABRAS_INTEGRADA = ("integrated", "integrada", "built-in", "facetime", "internal")

# Propiedades de dispositivo de Windows (devpkey.h): (fmtid, pid)
UBICACION_FISICA = ("540b947e-8b40-45bc-a8a2-6a0b894cbda2", 9)          # DEVPKEY_Device_PhysicalDeviceLocation
EN_CONTENEDOR_DEL_EQUIPO = ("8c7ed206-3f8a-4827-b3ab-ae9e1faefc6c", 4)  # DEVPKEY_Device_InLocalMachineContainer

if os.name == "nt":
    # Intentos para abrir la MISMA cámara integrada: (backend, nombre, pedir MJPG).
    # Los ajustes se pasan al abrir: con DirectShow + MJPG la cámara da 1280x720 a 30 FPS
    # y abre en ~0.5 s. Con set() después de abrir, Windows dejaba el formato YUY2
    # (sin comprimir), que por USB 2.0 solo llega a ~10 FPS en 720p.
    INTENTOS = ((cv2.CAP_DSHOW, "DirectShow", True),
                (cv2.CAP_MSMF, "Media Foundation", False),
                (cv2.CAP_DSHOW, "DirectShow", False))
else:
    INTENTOS = ((cv2.CAP_ANY, "por defecto", False),)
BACKENDS = tuple(dict.fromkeys((backend, nombre) for backend, nombre, _ in INTENTOS))


class CamaraNoDisponible(RuntimeError):
    pass


@dataclass
class Camara:
    indice: int        # índice para cv2.VideoCapture(indice, backend)
    backend: int
    nombre: str
    instancia: str     # ID de instancia de Windows, ej. USB\VID_04F2&PID_B7B9&MI_00\6&...
    integrada: bool
    detalle: str       # por qué se clasificó así
    puntaje: int = 0   # entre varias integradas se prefiere la frontal


def instancia_desde_ruta(ruta: str) -> str:
    r"""
    Convierte la ruta del dispositivo en su ID de instancia de Windows:
    \\?\usb#vid_04f2&pid_b7b9&mi_00#6&261941b5&1&0000#{guid}\global
        -> USB\VID_04F2&PID_B7B9&MI_00\6&261941B5&1&0000
    """
    if not ruta or not ruta.startswith("\\\\?\\"):
        return ""
    return ruta[4:].split("#{")[0].replace("#", "\\").upper()


def describir_ubicacion(pld: bytes):
    """Lee el byte de ubicación del buffer ACPI _PLD: (panel, está en la tapa)."""
    return (pld[8] >> 3) & 0x07, bool(pld[8] & 0x04)


class _DispositivosWindows:
    """Lee propiedades de dispositivos con la API de Windows (cfgmgr32)."""

    CR_EXITO, CR_BUFFER_PEQUENO = 0x00, 0x1A

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        class GUID(ctypes.Structure):
            _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                        ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]

        class DEVPROPKEY(ctypes.Structure):
            _fields_ = [("fmtid", GUID), ("pid", wintypes.ULONG)]

        self._ctypes = ctypes
        self._wt = wintypes
        self._GUID, self._DEVPROPKEY = GUID, DEVPROPKEY
        cfgmgr32 = ctypes.WinDLL("cfgmgr32")
        self._localizar = cfgmgr32.CM_Locate_DevNodeW
        self._localizar.argtypes = [ctypes.POINTER(wintypes.DWORD), wintypes.LPCWSTR, wintypes.ULONG]
        self._localizar.restype = wintypes.DWORD
        self._propiedad = cfgmgr32.CM_Get_DevNode_PropertyW
        self._propiedad.argtypes = [wintypes.DWORD, ctypes.POINTER(DEVPROPKEY), ctypes.POINTER(wintypes.ULONG),
                                    ctypes.c_void_p, ctypes.POINTER(wintypes.ULONG), wintypes.ULONG]
        self._propiedad.restype = wintypes.DWORD

    def _clave(self, clave):
        fmtid, pid = clave
        u = uuid.UUID(fmtid)
        guid = self._GUID(u.fields[0], u.fields[1], u.fields[2], (self._ctypes.c_ubyte * 8)(*u.bytes[8:]))
        return self._DEVPROPKEY(guid, pid)

    def leer(self, instancia: str, clave):
        """Devuelve los bytes de la propiedad, o None si el dispositivo no la tiene."""
        ct, wt = self._ctypes, self._wt
        nodo = wt.DWORD()
        if self._localizar(ct.byref(nodo), instancia, 0) != self.CR_EXITO:
            return None
        clave = self._clave(clave)
        tipo, tamano = wt.ULONG(), wt.ULONG(0)
        if self._propiedad(nodo, ct.byref(clave), ct.byref(tipo), None, ct.byref(tamano), 0) != self.CR_BUFFER_PEQUENO:
            return None
        datos = (ct.c_ubyte * tamano.value)()
        if self._propiedad(nodo, ct.byref(clave), ct.byref(tipo), datos, ct.byref(tamano), 0) != self.CR_EXITO:
            return None
        return bytes(datos[:tamano.value])


def clasificar(info, leer_propiedad=None) -> Camara:
    """Decide si una cámara (resultado de enumerate_cameras) es la integrada del equipo."""
    def camara(integrada, detalle, puntaje=0):
        return Camara(info.index, info.backend, info.name, instancia, integrada, detalle, puntaje)

    instancia = instancia_desde_ruta(info.path or "")
    if leer_propiedad is None:   # sin acceso a Windows: se decide por el nombre
        integrada = any(p in info.name.lower() for p in PALABRAS_INTEGRADA)
        return camara(integrada, "integrada (por su nombre)" if integrada else "no parece integrada",
                      1 if integrada else 0)

    if not instancia or instancia.split("\\")[0] in BUSES_VIRTUALES:
        return camara(False, "virtual (creada por un programa)")
    pld = leer_propiedad(instancia, UBICACION_FISICA)
    if pld and len(pld) > 8:
        panel, en_tapa = describir_ubicacion(pld)
        lugar = ("en la tapa, " if en_tapa else "") + f"panel {PANELES[panel]}"
        return camara(True, f"integrada ({lugar})", 3 if panel == PANEL_FRONTAL else 2)
    contenedor = leer_propiedad(instancia, EN_CONTENEDOR_DEL_EQUIPO)
    if contenedor and contenedor[0]:
        return camara(True, "integrada (parte interna del equipo)", 1)
    return camara(False, "externa (USB u otro dispositivo conectable)")


def listar_camaras(backend: int) -> list:
    """Todas las cámaras que ve OpenCV con ese backend, clasificadas. No abre ninguna."""
    if enumerate_cameras is None:
        raise CamaraNoDisponible("Falta la librería cv2-enumerate-cameras: pip install -r requirements.txt")
    lector = _DispositivosWindows().leer if os.name == "nt" else None
    return [clasificar(info, lector) for info in enumerate_cameras(backend)]


def elegir_integrada(camaras: list):
    """La cámara integrada con mayor puntaje (frontal primero); None si no hay ninguna."""
    integradas = [c for c in camaras if c.integrada]
    return max(integradas, key=lambda c: c.puntaje) if integradas else None


def _parametros(ancho: int, alto: int, fps: int, mjpg: bool) -> list:
    parametros = [cv2.CAP_PROP_FRAME_WIDTH, ancho, cv2.CAP_PROP_FRAME_HEIGHT, alto, cv2.CAP_PROP_FPS, fps]
    return [cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")] + parametros if mjpg else parametros


def abrir_camara_integrada(ancho: int, alto: int, fps: int = 30):
    """
    Abre la cámara integrada y devuelve (captura, camara). Nunca abre otra cámara:
    si la integrada no aparece o no se puede abrir, lanza CamaraNoDisponible.
    """
    listas = {}
    encontrada = None
    for backend, _, mjpg in INTENTOS:
        if backend not in listas:
            listas[backend] = listar_camaras(backend)
        camara = elegir_integrada(listas[backend])
        if camara is None:
            continue
        encontrada = camara
        captura = cv2.VideoCapture(camara.indice, camara.backend, _parametros(ancho, alto, fps, mjpg))
        if captura.isOpened() and captura.read()[0]:   # se comprueba que de verdad envía imágenes
            return captura, camara
        captura.release()

    if encontrada is not None:
        raise CamaraNoDisponible(
            f"La cámara integrada ({encontrada.nombre}) está ocupada o bloqueada.\n"
            "  - Cierra otras apps que la usen (Teams, Zoom, Meet, la app Cámara...).\n"
            "  - Revisa Configuración > Privacidad y seguridad > Cámara.")
    detectadas = "\n".join(f"  - {c.nombre}: {c.detalle}" for c in listar_camaras(BACKENDS[0][0]))
    raise CamaraNoDisponible(
        "No se encontró la cámara integrada del equipo (no se usará ninguna otra).\n"
        f"Cámaras detectadas:\n{detectadas or '  (ninguna)'}\n"
        "Revisa que la cámara integrada no esté desactivada (tecla o interruptor de privacidad\n"
        "del portátil, Administrador de dispositivos o Configuración > Privacidad > Cámara).")


def imprimir_camaras():
    """Muestra en consola cómo se clasificó cada cámara y cuál se usará."""
    for backend, nombre_backend in BACKENDS:
        camaras = listar_camaras(backend)
        elegida = elegir_integrada(camaras)
        print(f"Cámaras con {nombre_backend}:")
        for c in camaras:
            marca = "-> SE USA" if c is elegida else "   ignorada"
            print(f"  {marca}  [{c.indice}] {c.nombre}: {c.detalle}")
        if not camaras:
            print("  (ninguna)")
        print()
