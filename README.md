# Simulador de Agujero Negro tipo Kerr (Python + GPU)

Proyecto de visualización del campo gravitatorio de un **agujero negro tipo Kerr (aprox.)** usando **ray marching en la GPU** con Python, ModernGL y shaders GLSL.

- Autor: **Josafat Vásquez**

---

## Objetivo del proyecto

Desarrollar un simulador interactivo que permita:

- Visualizar un **agujero negro tipo Kerr** con:
  - Horizonte oscuro (sombra del agujero).
  - **Disco de acreción** en forma de aro delgado, brillante y anaranjado/rojizo.
- Mostrar un **planeta orbitando** alrededor del agujero negro.
- Observar efectos cualitativos de:
  - **Lente gravitacional**: deformación del planeta cuando pasa detrás del agujero.
  - **Arrastre de marco** (“frame dragging”) asociado al espín del agujero negro.
- Controlar la cámara en 3D con el mouse (rotar, trasladar, hacer “zoom”) manteniendo siempre la escala visual coherente aunque el observador se aleje o acerque.

---

## Descripción general

El proyecto está implementado en **Python** usando:

- `glfw` para crear la ventana y manejar el mouse.
- `moderngl` como wrapper de OpenGL moderno.
- `numpy` para operaciones vectoriales básicas en CPU.
- Shaders GLSL para hacer el **ray marching** y el sombreado en la GPU.

Estructura general:

- Se dibuja un **quad** de pantalla completa.
- En el **fragment shader**:
  - Se genera un rayo por píxel desde la cámara.
  - Se integra la trayectoria del rayo en un medio equivalente a una métrica tipo Schwarzschild con corrección de **espín** (Kerr aproximado).
  - Se detectan intersecciones con:
    - El horizonte del agujero negro.
    - El **disco de acreción** (aro en el plano \( y = 0 \) entre dos radios areales).
    - Un **planeta esférico** que orbita en el plano XZ.
  - Se calcula el color final incluyendo:
    - Emisión del disco (gradiente naranja–rojo).
    - Iluminación del planeta (difusa, especular y rim light).
    - Postprocesado tipo “film” (exposición, viñeta, sepia, grano, líneas de escaneo, polvo).

Además, el código ajusta dinámicamente la **escala física** (masa efectiva, radios del disco, órbita y tamaño del planeta) según la distancia de la cámara, para que:

- El agujero negro y el aro sigan viéndose bien tanto de cerca como de lejos.
- El comportamiento visual de la lente gravitacional sea estable incluso con mucho “zoom”.

---

## Fundamento numérico / físico (resumen)

El simulador no resuelve la métrica de Kerr exacta, pero implementa una aproximación basada en:

1. **Índice óptico equivalente (Schwarzschild isotrópico)**  
   Se modela el espacio-tiempo como un medio con índice de refracción efectivo $\( n(\rho) \)$.  
   A partir de una forma de la Ley de Fermat en relatividad, se puede escribir una aceleración óptica:

   $$\[
   \vec{a}_s \approx \nabla \ln n(\vec{r}) - \big( \vec{v} \cdot \nabla \ln n(\vec{r}) \big)\,\vec{v}
   \]$$

   donde:
   - $\( \vec{r} \)$ es la posición del rayo,
   - $\( \vec{v} \)$ es la dirección (velocidad unitaria) del rayo,
   - $\( \nabla \ln n \)$ se obtiene numéricamente en el shader.

2. **Corrección tipo Kerr (frame dragging)**  
   Se agrega un término de “arrastre” **azimutal**:

   $$\[
   \vec{a}_{fd} \propto \omega(r)\,\big( \hat{\varphi} - (\hat{\varphi}\cdot \vec{v}) \vec{v} \big)
   \]$$

   donde:
   - $\( \hat{\varphi} \)$ es el versor tangencial en el plano XZ,
   - $\( \omega(r) \)$ depende del espín adimensional $\( a \)$ y la masa $\( M \)$,
   - El parámetro `uA` (0–0.99) controla la intensidad de ese espín.

3. **Integración numérica (RK2)**  
   Para cada píxel se integra el rayo con un esquema tipo **Runge–Kutta de segundo orden**:

   - Se calcula la aceleración en el punto actual.
   - Se da un paso intermedio (midpoint).
   - Se corrige la dirección del rayo y se avanza en pasos de tamaño `uH`.

   Se corta la integración cuando:
   - El rayo cae dentro del horizonte (se considera “absorbido”).
   - Se aleja demasiado del sistema.
   - Se supera un número máximo de pasos.

4. **Intersección con disco y planeta**  
   Durante la integración se detectan:

   - **Disco de acreción**:  
     Intersección con el plano $\( y = 0 \)$. Se calcula el radio areal $\( r_A \)$ y se comprueba si está entre `uDiskInnerR` y `uDiskOuterR` (aro delgado).  
     El color se modula con:
     - Tono base naranja (`uDiskBaseCol`),
     - Región más caliente rojiza (`uDiskHotCol`),
     - Variación angular (estrías),
     - Intensidad global `uDiskIntensity`.

   - **Planeta**:  
     Intersección segmento–esfera (centro `uPlPos`, radio `uPlRadius`).  
     Se usa iluminación tipo Blinn–Phong + **rim light** para resaltar contorno y bandas sutiles según la normal.  
     Cuando el planeta pasa detrás del agujero negro, la trayectoria curvada de los rayos hace que su imagen se **distorsione** y se vea parcialmente alrededor del agujero (lente gravitacional).

---

## Requisitos

- **Python** 3.10 o superior.
- Librerías:
  - `glfw`
  - `moderngl`
  - `numpy`
- **GPU** con soporte para **OpenGL 3.3+** (recomendado: tarjeta dedicada; el código imprime el VENDOR y RENDERER en consola al iniciar).

Instalación de dependencias:

```bash
pip install glfw moderngl numpy
