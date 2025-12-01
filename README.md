Black Hole Kerr – GPU Ray Marching (Python + ModernGL)

Este proyecto es un experimento visual de un agujero negro tipo Kerr (aproximado) renderizado en la GPU mediante ray marching, usando Python, ModernGL y GLSL.
Incluye:

Agujero negro con disco de acreción tipo aro (delgado, brillante y anaranjado/rojizo).

Un planeta que orbita alrededor del agujero negro.

Lente gravitacional que deforma la imagen del planeta cuando pasa por detrás.

Cámara libre controlada con el mouse.

Look “vintage” opcional (sepia, grano, líneas de escaneo, polvo, viñeta, etc.).

Todo el cálculo de la luz (geodésicas ópticas en una métrica tipo Schwarzschild + término de arrastre tipo Kerr) se realiza en el fragment shader, aprovechando la GPU.

Requisitos

Python 3.10 o superior

Librerías de Python:

glfw

moderngl

numpy

GPU compatible con OpenGL 3.3 o superior

Instalación de dependencias:

pip install glfw moderngl numpy

Cómo ejecutar

Guarda el archivo Python (por ejemplo BlackHole.py) con el código del proyecto.

Ejecuta desde la terminal:

python BlackHole.py


Al iniciar, la ventana mostrará:

Un agujero negro en el centro con un aro de disco de acreción.

Un planeta orbitando, que se deforma visualmente por la curvatura del espacio-tiempo cuando pasa por detrás del agujero negro.

Información de la GPU usada (vendor, renderer, versión OpenGL) impresa en consola.

Controles

La cámara es completamente libre y se controla solo con el mouse:

Rotar cámara:
Clic derecho sostenido + mover mouse.

Moverse en el plano (strafe / adelante-atrás):
Clic izquierdo sostenido + mover mouse.

Subir / bajar:
Clic central (rueda) sostenido + mover mouse verticalmente.

Zoom (dolly hacia/desde el agujero):
Rueda del mouse:

Scroll hacia delante: acercarse al agujero negro.

Scroll hacia atrás: alejarse.

El sistema adapta la escala física (masa, radios del disco, órbita y tamaño del planeta) según la distancia de la cámara, de modo que:

El agujero negro, el aro y la órbita se siguen viendo bien tanto de cerca como desde lejos.

El comportamiento visual de la lente gravitacional se mantiene estable aunque hagas mucho zoom.

Arquitectura del código

El proyecto se compone de dos partes principales:

Código Python (CPU)

Configura la ventana con GLFW.

Crea el contexto OpenGL usando ModernGL.

Compila los shaders (VERT y FRAG).

Crea un quad de pantalla completa para dibujar (dos triángulos).

Controla la cámara (posición, yaw, pitch, zoom) y gestiona los eventos del mouse.

Calcula parámetros físicos escalados según la distancia de la cámara:

Masa efectiva M

Radios interno/externo del disco

Radio de la órbita

Tamaño del planeta

Actualiza los uniforms del shader en cada frame.

Shaders GLSL (GPU)

Vertex shader:
Solo pasa un quad de pantalla completa y genera coordenadas de UV.

Fragment shader:

Construye el rayo inicial a partir de la cámara, FOV y resolución de pantalla.

Integra la trayectoria del rayo en un medio con índice óptico equivalente a Schwarzschild, con un término adicional tipo Kerr para el arrastre de marco.

Detecta:

Cruce con el horizonte del agujero negro (esfera “oscura”).

Cruce con el plano del disco de acreción (aro entre radios uDiskInnerR y uDiskOuterR).

Cruce con el planeta (intersección segmento–esfera).

Calcula el color del disco:

Aro delgado con gradiente anaranjado/rojo.

Variación azimutal (estrías) y reforzamiento en la zona interior (rim).

Calcula el color del planeta:

Iluminación difusa (Blinn–Phong), especular y rim light para silueta.

Ligeras bandas de color según la “latitud” en la esfera.

Aplica postprocesado:

Exposición tipo “filmic”.

Bloom falso.

Lift / gamma / gain.

Sepia, grano, líneas de escaneo, polvo, viñeta, tintado en bordes.

Parámetros importantes que puedes ajustar

En la función main() (lado Python) están los parámetros de alto nivel:

Escala física y FOV:

fov_deg     = 90.0
base_steps  = 800.0
base_hstep  = 0.05
base_M      = 1.0
aSpin       = 0.6


Disco (aro del agujero negro):

base_disk_inner = 9 * base_M       # radio interno (en unidades de M)
base_disk_outer = 10 * base_M      # radio externo
disk_base  = (1.0, 0.45, 0.02)     # naranja
disk_hot   = (1.0, 0.08, 0.02)     # rojo
disk_int   = 2.1                   # intensidad global


Para:

Aro más delgado: acercar base_disk_outer a base_disk_inner.

Aro más grande: aumentar ambos radios manteniendo el grosor parecido.

Aro más brillante: aumentar disk_int.

Planeta y órbita:

base_pl_radius = 2.2
base_orbit_r   = 16.0 * base_M
orbit_w        = 0.16  # velocidad angular


Más lejos: subir base_orbit_r.

Más rápido: subir orbit_w.

Más grande: subir base_pl_radius.

Look vintage:

sepia   = 0.35
grain   = 0.40
scan    = 0.28
flicker = 0.15
jitter  = 0.15
dust    = 0.25
edge_t  = 0.45


Si quieres un look más limpio (más “científico”), puedes bajar estos valores a casi cero.

Ideas para extensiones futuras

Añadir controles con teclado:

Resetear cámara.

Alternar look vintage on/off en tiempo real.

Congelar/retomar la órbita del planeta.

Agregar más planetas o anillos secundarios.

Implementar un disco de acreción más físico (perfil de brillo y color según la distancia al agujero).

Exportar capturas de pantalla con una tecla dedicada.

# Para uso personal/educativo (según lo necesites)
