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
s_radius = res / 15.0

# Estructuras de datos
## Campos de densidad
# _density_field_1 = ti.field(float, shape=(res, res))
# _density_field_2 = ti.field(float, shape=(res, res))
## Campos de densidad con nuevo tamaño de grid
_density_field_1 = ti.field(float, shape=(N, N))
_density_field_2 = ti.field(float, shape=(N, N))

dens = FieldPair(_density_field_1, _density_field_2)

# Campos de velocidad
# vel_u = ti.field(float, shape=(res, res))
# vel_v = ti.field(float, shape=(res, res))
# campos de velocidad con nuevo tamaño de grid
vel_u = ti.field(float, shape=(N, N))
vel_v = ti.field(float, shape=(N, N))


@ti.kernel
def add_sources(dens: ti.template(), input_data: ti.types.ndarray(),vel_u: ti.template(),vel_v: ti.template()):
    for i, j in dens:
        densidad = input_data[2] * s_dens
        mx, my = input_data[0], input_data[1]
        # Centro casilla i,j (una casilla 1pixel->centro=0.5)
        cx = i + 0.5
        cy = j + 0.5
        # Distancia al centro de la casilla (al cuadrado)
        d2 = (cx - mx) ** 2 + (cy - my) ** 2 #pitagoras
        if input_data[2] != 0:
            # Decaimiento exponencial
            if d2 < s_radius:
                dens[i, j] += dt * densidad * ti.exp(-6 * d2 / s_radius**2)

        if input_data[5] != 0:
                if d2 < s_radius:
                    vel_u[i, j] += input_data[3] * ti.exp(-6 * d2 / s_radius**2)
                    vel_v[i, j] += input_data[4] * ti.exp(-6 * d2 / s_radius**2)
    

def init():
    dens.cur.fill(0)
    dens.nxt.fill(0)
    init_vel()

@ti.kernel  
def init_vel():
    for i, j in ti.ndrange((1, res+1), (1, res+1)):
        #Inicializar las velocidades a cero
        vel_u[i, j] = 0.0
        vel_v[i, j] = 0.0
        



def step(input_data):
    add_sources(dens.cur, input_data, vel_u, vel_v)
    if(input_data[3] != 0 or input_data[4] != 0):
        debug_velocities()
    set_boundaries(dens.cur)
    for _ in range(100): 
        diffuse_jacobi(dens.cur, dens.nxt)
        set_boundaries(dens.nxt)
        dens.swap()
    adveccion(dens.cur, dens.nxt)
    set_boundaries(dens.nxt)
    dens.swap()
    
@ti.kernel
def debug_velocities():
    for i, j in vel_u:
        if vel_u[i, j] != 0.0 or vel_v[i, j] != 0.0:
            print("i:", i, "j:", j, "vel_u:", vel_u[i, j], "vel_v:", vel_v[i, j])
    
      

@ti.kernel
def diffuse_jacobi(cur: ti.template(), nxt: ti.template()):
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
    #version sin set_boundaries
    # for i, j in ti.ndrange(res, res):
    #     if 0 < i < res-1 and 0 < j < res-1:
    #         nxt[i, j] = ((cur[i, j] + alpha * (cur[i-1, j] + cur[i+1, j] + cur[i, j-1] + cur[i, j+1]))/(1+4*alpha)) 
    #     else:
    #         nxt[i, j] = cur[i, j]
    
    #nueva version: version con set_boundaries
    for i, j in ti.ndrange((1, res+1), (1, res+1)):
        nxt[i, j] = ((cur[i, j] + alpha * (cur[i-1, j] + cur[i+1, j] + cur[i, j-1] + cur[i, j+1]))/(1+4*alpha)) 
       



    
@ti.kernel
def adveccion(cur: ti.template(), nxt: ti.template()):
    dt0 = dt * res
    for i, j in ti.ndrange((1, res+1), (1, res+1)):
        x = i - dt0 * vel_u[i, j]
        y = j - dt0 * vel_v[i, j]
        nxt[i, j] = bilerp(cur, ti.Vector([x, y]))

@ti.kernel
def set_boundaries(dens: ti.template()):
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

@ti.func
def debug(x,y):
    print("valor:", x, y) 

@ti.func
def debugtxt(txt):
    print(txt) 
    
@ti.func
def bilerp(vf, p):
    u, v = p
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
    I = ti.max(0, ti.min(res - 1, I))
    return qf[I]

def main():
    paused = False
    window = ti.ui.Window("Stable Fluids", (res, res), vsync=True)
    canvas = window.get_canvas()

    # Inicialización
    init()
    prev_mouse = None #posicion anterior del raton para calcular el movimiento

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
            step(input_data)
        else:
            print("pausado")

        # Renderizado
        canvas.set_image(dens.cur)
        window.show()


if __name__ == "__main__":
    main()
