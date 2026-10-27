# Métodos numéricos - Proyecto final: simulación de fluidos 2D

**Simulador interactivo de fluidos en 2D basado en el método *Stable Fluids*, implementado en Python con Taichi para ejecutarse en GPU.**

Es el proyecto final de la asignatura de Métodos Numéricos del máster. El simulador se construye de forma incremental en **cinco etapas**: cada una añade un fenómeno físico nuevo hasta obtener un fluido incompresible con el que se puede interactuar con el ratón. Con el botón derecho se añade densidad (el «humo») y con el izquierdo se arrastra para aplicar fuerzas al fluido.



## Enlaces

- 🎥 **Vídeo de demostración:** [youtu.be/Bo4WJFE8wgI](https://youtu.be/Bo4WJFE8wgI)

## Etapas del proyecto

| Etapa | Qué añade | Método numérico |
|---|---|---|
| **1. Difusión** | Fuentes de densidad con el ratón que se difunden por el grid | Iteraciones de Jacobi |
| **2. Advección** | La densidad se transporta con un campo de velocidades fijo de tres franjas horizontales | Método semi-Lagrangiano con interpolación bilineal |
| **3. Fuerzas externas** | El usuario empuja el fluido arrastrando el ratón | Fuerza con decaimiento exponencial |
| **4. Dinámica de la velocidad** | La velocidad también se difunde y se advecta a sí misma, con paredes que rebotan el fluido | Jacobi, semi-Lagrangiano y fronteras reflectivas |
| **5. Incompresibilidad** | El campo de velocidades se proyecta para que no tenga divergencia, lo que genera los remolinos característicos | Ecuación de Poisson resuelta con Gauss-Seidel |

## Detalle de cada etapa

### Etapa 1: Difusión (`etapa1.py`)

- **Grid con bordes:** la simulación trabaja sobre un grid de 512 × 512 celdas, más una fila y una columna extra en cada lado para las condiciones de frontera.
- **Fuentes de densidad:** al pulsar el ratón se añade densidad alrededor del cursor con un decaimiento exponencial según la distancia al centro de cada celda.
- **Difusión estable con Jacobi:** la ecuación de difusión explícita es inestable cuando k·dt/h² > 0,5. Para evitarlo se usa la aproximación de Jacobi, que da una solución estable con cualquier paso de tiempo.
- **Condiciones de frontera:** en los bordes se copia el valor de la celda adyacente y en las esquinas se usa el promedio de los vecinos.
- **Doble búfer:** la clase `FieldPair` guarda el estado actual y el siguiente de cada campo y los intercambia en cada paso sin copiar datos.

### Etapa 2: Advección (`etapa2.py`)

- Se añade un campo de velocidades inicializado en **tres franjas horizontales** que se mueven en sentidos alternos.
- La densidad se transporta con el **método semi-Lagrangiano hacia atrás**: para cada celda se calcula de dónde vendría la partícula en el paso anterior y se interpola su valor con una **interpolación bilineal**.

### Etapa 3: Fuerzas externas (`etapa3.py`)

- El campo de velocidades empieza a cero y el usuario lo modifica arrastrando el ratón con el botón izquierdo.
- El desplazamiento del cursor entre fotogramas se convierte en una fuerza que se aplica alrededor del cursor, también con decaimiento exponencial.

### Etapa 4: Dinámica de la velocidad (`etapa4.py`)

- Se añade un paso de velocidad (`vel_step`) en el que **la velocidad también se difunde y se advecta a sí misma**, antes de actualizar la densidad.
- **Fronteras reflectivas:** en cada pared se copia la componente tangencial de la velocidad y se invierte la normal, para que el fluido rebote en lugar de salir del dominio.
- Los campos de velocidad también usan `FieldPair` para alternar entre el estado actual y el anterior sin copias.

### Etapa 5: Proyección e incompresibilidad (`etapa5.py`)

La última etapa hace que el fluido sea **incompresible**, para que la masa se conserve y aparezcan remolinos realistas. La proyección se aplica dentro del paso de velocidad:

1. Se calcula la **divergencia** del campo de velocidades.
2. Se resuelve la **ecuación de Poisson** para la presión con **50 iteraciones de Gauss-Seidel**.
3. Se **resta el gradiente de presión** a la velocidad, con lo que se obtiene un campo sin divergencia.

## Tecnologías

- **Python 3.13**
- **Taichi** para ejecutar los kernels de simulación en GPU (Vulkan o CUDA) y para la ventana interactiva (GGUI)
- **NumPy** para pasar los datos de entrada del ratón a los kernels

## Recursos utilizados

- Apuntes y explicaciones vistos en clase

## Contenido del repositorio

- `etapa1.py` a `etapa5.py`: una versión del simulador por etapa. Cada archivo amplía el anterior.

## Autora

Olaia Picabea
