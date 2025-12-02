# Black Hole Kerr – GPU (Python + ModernGL)

- Este proyecto es un experimento visual de un agujero negro tipo Kerr (aproximado) renderizado en la GPU mediante ray marching, usando Python, ModernGL y GLSL.
- Este proyecto puede usarse con fines personales y educativos.

Incluye:

- Agujero negro con disco de acreción tipo aro (delgado, brillante y anaranjado/rojizo).
- Un planeta que orbita alrededor del agujero negro.
- Lente gravitacional que deforma la imagen del planeta cuando pasa por detrás del agujero negro.
- Cámara libre controlada con el mouse.
- Look “vintage” opcional (sepia, grano, líneas de escaneo, polvo, viñeta, tintado de bordes).

Todo el cálculo de la luz (geodésicas ópticas en una métrica tipo Schwarzschild con término de arrastre tipo Kerr) se realiza en el fragment shader, aprovechando la GPU.

---

## Requisitos

- Python 3.10 o superior
- Librerías de Python:
  - `glfw`
  - `moderngl`
  - `numpy`
- GPU compatible con OpenGL 3.3 o superior

Instalación de dependencias:

```bash
pip install glfw moderngl numpy
