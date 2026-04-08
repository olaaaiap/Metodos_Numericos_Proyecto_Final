# Referencia: https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/ggui_examples/stable_fluid_ggui.py

import numpy as np
import taichi as ti

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)
# ti.init(arch=ti.cpu, debug=True)


class FieldPair:
    def __init__(self, current_field, next_field):
        self.cur = current_field
        self.nxt = next_field

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


# Parametros de la simulacion de fluidos
res = 512  # Resolución del grid
N = res + 2 #Nuevo tamaño de grid (con bordes)
h = 1 / res  # Tamaño de la celda
dt = 0.03  # Tamaño del paso de tiempo

## Parametros de la fuente de densidad
s_dens = 10.0
s_fuerza = 5.0
s_radius = res / 15.0

# Estructuras de datos
## Campos de densidad con nuevo tamaño de grid
_density_field_1 = ti.field(float, shape=(N, N))
_density_field_2 = ti.field(float, shape=(N, N))

dens = FieldPair(_density_field_1, _density_field_2)

# campos de velocidad con nuevo tamaño de grid
vel_u_1 = ti.field(float, shape=(N, N))
vel_v_1 = ti.field(float, shape=(N, N))
# variables para guardar el estado anterior
vel_u0 = ti.field(float, shape=(N, N))
vel_v0 = ti.field(float, shape=(N, N))
#Convertir a Fieldpair para no tener que copiar vel_u a vel_u0
vel_u = FieldPair(vel_u_1, vel_u0)
vel_v = FieldPair(vel_v_1, vel_v0)


divergence = ti.field(float, shape=(N, N))
pressure = ti.field(float, shape=(N, N))


@ti.kernel
def add_sources(dens: ti.template(), input_data: ti.types.ndarray(),vel_u: ti.template(),vel_v: ti.template()):
    for i, j in ti.ndrange((1, res+1), (1, res+1)):
        densidad = input_data[2] * s_dens
        mx, my = input_data[0], input_data[1]
        # Centro casilla i,j (una casilla 1pixel->centro=0.5)
        cx = i + 0.5
        cy = j + 0.5
        # Distancia al centro de la casilla (al cuadrado)
        d2 = (cx - mx) ** 2 + (cy - my) ** 2 #pitagoras
        if input_data[2] != 0:
            # Decaimiento exponencial
            if d2 < s_radius**2: 
                dens[i, j] += dt * densidad * ti.exp(-6 * d2 / s_radius**2)

        if input_data[5] != 0:
            if d2 < s_radius**2:
                vel_u[i, j] += input_data[3] * ti.exp(-6 * d2 / s_radius**2) * s_fuerza
                vel_v[i, j] += input_data[4] * ti.exp(-6 * d2 / s_radius**2) * s_fuerza
    

def init():
    dens.cur.fill(0)
    dens.nxt.fill(0)
    vel_u.cur.fill(0.0)
    vel_u.nxt.fill(0.0)
    vel_v.cur.fill(0.0)
    vel_v.nxt.fill(0.0)


    
def vel_step(input_data):
    """
    Nueva función step para aplicar la difusión y advección a las velocidades y a las densidades.

    :param input_data: Datos de entrada
    """
    add_sources(dens.cur, input_data, vel_u.cur, vel_v.cur)
    set_boundaries_reflective(vel_u.cur, vel_v.cur)

    #Aplicar difusión a las velocidades
    for _ in range(20): 
        jacobi_iter(vel_u.cur, vel_u.nxt)
        jacobi_iter(vel_v.cur, vel_v.nxt)
        set_boundaries_reflective(vel_u.nxt, vel_v.nxt)

        vel_u.swap()
        vel_v.swap()
        set_boundaries_reflective(vel_u.cur, vel_v.cur) 
        
    #Aplicar la proyección para obtener un campo de velocidades incompresible
    proyeccion(vel_u.cur, vel_v.cur, pressure, divergence)

    #Aplicar la advección a las velocidades
    advect(vel_u.cur, vel_u.nxt, vel_u.cur, vel_v.cur)
    advect(vel_v.cur, vel_v.nxt, vel_u.cur, vel_v.cur)
    set_boundaries_reflective(vel_u.nxt, vel_v.nxt)

    vel_u.swap()
    vel_v.swap()
    set_boundaries_reflective(vel_u.cur, vel_v.cur) 
    proyeccion(vel_u.cur, vel_v.cur, pressure, divergence)

    #step anterior. calculo de la densidad
    step()

def step():
    diffuse(dens)
    advect(dens.cur, dens.nxt, vel_u.cur, vel_v.cur)
    set_boundaries(dens.nxt)
    dens.swap()
    

def proyeccion(vel_u: ti.template(), vel_v: ti.template(), pressure: ti.template(), divergence: ti.template()):
    """
    Función para proyectar el campo de velocidades y obtener un campo incompresible.

    :param vel_u: Campo de velocidad horizontal
    :param vel_v: Campo de velocidad vertical
    :param pressure: Campo de presión
    :param divergence: Campo de divergencia
    """
    
    calcular_divergencia(vel_u, vel_v, divergence, pressure)

    set_boundaries(divergence)
    set_boundaries(pressure)

    
    for _ in range(50):
        resolver_poisson_gauss_seidel(pressure, divergence)
        set_boundaries(pressure)
    restar_gradiente(vel_u, vel_v, pressure)


    set_boundaries_reflective(vel_u, vel_v)

@ti.kernel
def resolver_poisson_gauss_seidel(pressure: ti.template(), divergence: ti.template()):
    """
    Función para resolver la ecuación de Poisson utilizando el método de Gauss-Seidel.
    
    :param pressure: Campo de presión
    :param divergence: Campo de divergencia
    """
    for i, j in ti.ndrange((1, res+1), (1, res+1)):
        pressure[i, j] = (
            divergence[i, j]
            + pressure[i-1, j] + pressure[i+1, j]
            + pressure[i, j-1] + pressure[i, j+1]
        ) / 4.0

@ti.kernel
def calcular_divergencia(vel_u: ti.template(), vel_v: ti.template(), divergence: ti.template(), pressure: ti.template()):
    """
    Función para calcular la divergencia del campo de velocidades y preparar el campo de presión para la proyección.

    :param vel_u: Campo de velocidad horizontal
    :param vel_v: Campo de velocidad vertical
    :param divergence: Campo de divergencia
    :param pressure: Campo de presión
    """
    for i, j in ti.ndrange((1, res+1), (1, res+1)):
        divergence[i, j] = -0.5 * h * (vel_u[i + 1, j] - vel_u[i - 1, j] + vel_v[i, j + 1] - vel_v[i, j - 1])
        pressure[i, j] = 0.0


@ti.kernel
def restar_gradiente(vel_u: ti.template(), vel_v: ti.template(), pressure: ti.template()):
    
    """
    Función para restar el gradiente de presión al campo de velocidades, obteniendo así un campo de velocidades incompresible.
    :param vel_u: Campo de velocidad horizontal
    :param vel_v: Campo de velocidad vertical
    :param pressure: Campo de presión
    """

    for i, j in ti.ndrange((1, res+1), (1, res+1)):
        vel_u[i, j] -= 0.5 * (pressure[i + 1, j] - pressure[i - 1, j]) / h
        vel_v[i, j] -= 0.5 * (pressure[i, j + 1] - pressure[i, j - 1]) / h

def diffuse(dens):
    for _ in range(20): 
        jacobi_iter(dens.cur, dens.nxt)
        set_boundaries(dens.nxt)
        dens.swap()


@ti.kernel
def jacobi_iter(cur: ti.template(), nxt: ti.template()):
    """
    Función para calcular la difusión utilizando el método de Jacobi.
    La ecuación de difusión es inestable cuando k*dt/h**2 > 0.5. 
    Para solucionar el problema se aplica la aproximación de Jacobi, donde en vez de calcular el laplaciano con nxt, 
    se calcula con cur, así obteniendo una solución estable.

    :param cur: Campo de densidad actual
    :param nxt: Campo de densidad siguiente
    """
    k = 1e-4
    alpha = k*dt/(h*h)
    
    for i, j in ti.ndrange((1, res+1), (1, res+1)):
        nxt[i, j] = ((cur[i, j] + alpha * (cur[i-1, j] + cur[i+1, j] + cur[i, j-1] + cur[i, j+1]))/(1+4*alpha)) 
       

 
@ti.kernel
def advect(cur: ti.template(), nxt: ti.template(), vel_u_cur: ti.template(), vel_v_cur: ti.template()):
    """
    Función para calcular la advección utilizando el método de semi-Lagraniano (backwards). 

    :param cur: Campo de densidad actual
    :param nxt: Campo de densidad siguiente
    :param vel_u_cur: Campo de velocidad horizontal actual
    :param vel_v_cur: Campo de velocidad vertical actual
    """
    dt0 = dt * res
    for i, j in ti.ndrange((1, res+1), (1, res+1)):
        x = (i + 0.5) - dt0 * vel_u_cur[i, j]
        y = (j + 0.5) - dt0 * vel_v_cur[i, j]

        nxt[i, j] = bilerp(cur, ti.Vector([x, y]))

@ti.kernel
def set_boundaries(dens: ti.template()):
    """
    Función para establecer las condiciones de frontera en el campo de densidad.
    Se copian los valores de las celdas adyacentes para los bordes y se calcula el promedio de
    los valores de los vecinos para las esquinas.

    :param dens: Campo de densidad
    """
    for i in range(1, res + 1):
        #para los bordes horizontales...
        dens[i, 0] = dens[i, 1] #...copiar el valor de la siguiente fila
        dens[i, res+1] = dens[i, res]#...copiar el valor de la anterior fila
        #para los bordes verticales...
        dens[0, i] = dens[1, i]#...copiar el valor de la siguiente columna
        dens[res+1, i] = dens[res, i] #...copiar el valor de la anterior columna
        
    #Esquinas: promedio de los vecinos
    dens[0, 0]= 0.5*(dens[1, 0] + dens[0, 1])
    dens[0, res+1] = 0.5*(dens[1, res+1] + dens[0, res])
    dens[res+1, 0] = 0.5*(dens[res, 0] + dens[res+1, 1])
    dens[res+1, res+1] = 0.5*(dens[res, res+1] + dens[res+1, res])

@ti.kernel
def set_boundaries_reflective(vel_u_nxt: ti.template(), vel_v_nxt: ti.template()):
    """
    Función para establecer las condiciones de frontera reflectivas en los campos de velocidad.
    Para los bordes horizontales se copia la componente tangencial y se invierte la normal, y para los bordes verticales se hace lo mismo pero con la componente vertical. 
    Para las esquinas se calcula el promedio de los vecinos.
    
    :param vel_u_nxt: Campo de velocidad horizontal
    :param vel_v_nxt: Campo de velocidad vertical
    """
    for i in range(1, res + 1):
        #para los bordes horizontales...
        vel_u_nxt[i, 0] = vel_u_nxt[i, 1] #...copiar la tangencial
        vel_v_nxt[i, 0] = -vel_v_nxt[i, 1]#...invertir el valor

        vel_u_nxt[i, res+1] = vel_u_nxt[i, res]#...copiar la tangencial
        vel_v_nxt[i, res+1] = -vel_v_nxt[i, res]#...invertir el valor


    for i in range(1, res + 1):
        #para los bordes verticales...
        vel_u_nxt[0,i] = -vel_u_nxt[1, i]#...invertir el valor
        vel_v_nxt[0, i] = vel_v_nxt[1, i]#...copiar la tangencial
        vel_u_nxt[res+1, i] = -vel_u_nxt[res, i]#...invertir el valor
        vel_v_nxt[res+1, i] = vel_v_nxt[res, i] #...copiar la tangencial
        
    
    vel_u_nxt[0, 0] = 0.5*(vel_u_nxt[1,0] + vel_u_nxt[0,1])
    vel_v_nxt[0, 0] = 0.5*(vel_v_nxt[1,0] + vel_v_nxt[0,1])

    vel_u_nxt[0, res+1] = 0.5*(vel_u_nxt[1, res+1] + vel_u_nxt[0, res])
    vel_v_nxt[0, res+1] = 0.5*(vel_v_nxt[1, res+1] + vel_v_nxt[0, res])

    vel_u_nxt[res+1, 0] = 0.5*(vel_u_nxt[res, 0] + vel_u_nxt[res+1, 1])
    vel_v_nxt[res+1, 0] = 0.5*(vel_v_nxt[res, 0] + vel_v_nxt[res+1, 1])

    vel_u_nxt[res+1, res+1] = 0.5*(vel_u_nxt[res, res+1] + vel_u_nxt[res+1, res])
    vel_v_nxt[res+1, res+1] = 0.5*(vel_v_nxt[res, res+1] + vel_v_nxt[res+1, res])


@ti.func
def bilerp(vf, pressure):
    u, v = pressure
    s, t = u - 0.5, v - 0.5
    # floor
    iu, iv = ti.floor(s), ti.floor(t)
    # fract
    fu, fv = s - iu, t - iv
    a = sample(vf, iu, iv)
    b = sample(vf, iu + 1, iv)
    c = sample(vf, iu, iv + 1)
    d = sample(vf, iu + 1, iv + 1)
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)


@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)


@ti.func
def sample(qf, u, v):
    I = ti.Vector([int(u), int(v)])
    I = ti.max(1, ti.min(res + 1, I))
    return qf[I]

def main():
    paused = False
    window = ti.ui.Window("Stable Fluids", (res, res), vsync=True)
    canvas = window.get_canvas()

    # Inicialización
    init()
    prev_mouse = None

    # Bucle Principal
    while window.running:
        # 0: mouse_x 1: mouse_y 2: source_active 3: fuerza_x 4: fuerza_y 5: fuerza_activa
        input_data = np.zeros(6, dtype=np.float32)

        # Input Teclado
        if window.get_event(ti.ui.PRESS):
            e = window.event
            if e.key == ti.ui.ESCAPE:  # ESC para salir
                break
            elif e.key == "r":  # 'r' para resetear
                paused = False
                init()
            elif e.key == "p":  # 'p' para pausar/reanudar
                paused = not paused

        mouse_xy = window.get_cursor_pos()
        input_data[0] = mouse_xy[0] * res
        input_data[1] = mouse_xy[1] * res

        # Input Ratón
        if window.is_pressed(ti.ui.RMB):
            input_data[2] = 1.0

        if window.is_pressed(ti.ui.LMB):
            input_data[5] = 1.0
            if prev_mouse is not None:
                input_data[3] = (mouse_xy[0] - prev_mouse[0]) * res
                input_data[4] = (mouse_xy[1] - prev_mouse[1]) * res
            prev_mouse = mouse_xy
        else:
            prev_mouse = None


        # Simulación (siguiente paso de tiempo)
        if not paused:
            # step(input_data)
            vel_step(input_data)    
        else:
            print("pausado")

        # Renderizado
        canvas.set_image(dens.cur)
        window.show()



if __name__ == "__main__":
    main()
