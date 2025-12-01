# Agujero negro tipo Kerr 
# Requisitos: Python 3.10+, glfw, moderngl, numpy

import glfw, moderngl, numpy as np, time, math

# ================== Shaders ==================
VERT = """
#version 330
in vec2 in_pos;
out vec2 uv;
void main(){
    uv = in_pos*0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRAG = """
#version 330
precision highp float;

uniform vec2  iResolution;
uniform float uTime;

uniform float uFov;          // rad
uniform float uSteps;        // pasos de integración
uniform float uH;            // tamaño de paso
uniform float uM;            // masa (G=c=1)
uniform float uA;            // espín (0..0.99)
uniform float uExposure;     // exposición global

// Look vintage / film
uniform float uSepia;        // 0..1
uniform float uGrain;        // 0..1
uniform float uScan;         // 0..1
uniform float uFlicker;      // 0..1
uniform float uJitter;       // 0..1
uniform float uDust;         // 0..1
uniform float uEdgeTint;     // 0..1

// Grading simple
uniform vec3  uLift;         // -0.1..0.3
uniform float uGamma;        // 0.6..1.6 (1/gamma)
uniform vec3  uGain;         // 0.7..1.5

// Cámara
uniform vec3  uCamPos;
uniform vec3  uCamFwd;
uniform vec3  uCamRight;
uniform vec3  uCamUp;

// Planeta
uniform vec3  uPlPos;        // centro esfera
uniform float uPlRadius;     // radio
uniform vec3  uPlAlbedo;     // color difuso del planeta
uniform vec3  uSunDir;       // luz direccional (normalizada)
uniform float uPlSpecular;   // fuerza especular (0..1)
uniform float uPlShininess;  // exponente especular (16..128)
uniform float uPlAmbient;    // luz ambiente base (0..1)
uniform vec3  uRimColor;     // rim light
uniform float uRimStrength;  // 0..1

// Disco de acreción (aro)
uniform float uDiskInnerR;   // radio interno (areal)
uniform float uDiskOuterR;   // radio externo (areal)
uniform vec3  uDiskBaseCol;  // color base (naranja)
uniform vec3  uDiskHotCol;   // color caliente (rojo)
uniform float uDiskIntensity;// intensidad global

in vec2 uv;
out vec4 fragColor;

float lenSafe(vec3 v){ return max(length(v), 1e-8); }

// ---- Schwarzschild (isotrópico): índice óptico ----
float dlogn_drho(float rho, float M){
    float a = M / (2.0*max(rho, 1e-6));
    float da = -M / (2.0*rho*rho);
    float term = 3.0/(1.0+a) + 1.0/(1.0-a);
    return term * da;
}
vec3 grad_logn(vec3 x, float M){
    float rho = lenSafe(x);
    float dl = dlogn_drho(rho, M);
    return dl * (x / rho);
}

// ---- Aproximación Kerr: frame-dragging ----
float rho_to_areal(float rho, float M){
    float a = M / (2.0*max(rho, 1e-6));
    return rho * (1.0 + a)*(1.0 + a);
}
vec3 phi_hat(vec3 x){
    float r = length(x.xz);
    if(r < 1e-6) return vec3(0.0);
    return normalize(vec3(-x.z, 0.0, x.x));
}

// --- Utils vintage
float hash12(vec2 p){
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
vec3 apply_sepia(vec3 c, float k){
    vec3 s;
    s.r = dot(c, vec3(0.393, 0.769, 0.189));
    s.g = dot(c, vec3(0.349, 0.686, 0.168));
    s.b = dot(c, vec3(0.272, 0.534, 0.131));
    return mix(c, s, k);
}
vec3 apply_scanlines(vec3 c, vec2 fragUV, float k){
    if(k<=0.0) return c;
    float line = 0.5 + 0.5*cos(2.0*3.14159 * iResolution.y * fragUV.y);
    return c * mix(1.0, 0.86 + 0.14*line, k);
}
vec3 apply_grain(vec3 c, vec2 fragUV, float t, float k){
    if(k<=0.0) return c;
    float n = hash12(fragUV * iResolution + t*120.0);
    n = n*2.0 - 1.0;
    return c + k * n * 0.03;
}
vec2 jitter_uv(vec2 u, float t, float k){
    if(k<=0.0) return u;
    float jx = (hash12(vec2(t, 0.37)) - 0.5) * 0.003 * k;
    float jy = (hash12(vec2(0.19, t)) - 0.5) * 0.003 * k;
    return clamp(u + vec2(jx, jy), 0.0, 1.0);
}

// Intersección segmento-esfera (xPrev->xNow) con esfera (C,R)
bool sphere_cross_segment(vec3 xPrev, vec3 xNow, vec3 C, float R, out float tSeg, out vec3 hitP, out vec3 hitN){
    vec3 d = xNow - xPrev;
    vec3 m = xPrev - C;
    float a = dot(d,d);
    float b = 2.0*dot(m,d);
    float c = dot(m,m) - R*R;

    float disc = b*b - 4.0*a*c;
    if(disc < 0.0){ tSeg = -1.0; return false; }

    float sdisc = sqrt(max(disc, 0.0));
    float t1 = (-b - sdisc) / (2.0*a);
    float t2 = (-b + sdisc) / (2.0*a);

    float tCand = -1.0;
    if(t1>=0.0 && t1<=1.0) tCand = t1;
    else if(t2>=0.0 && t2<=1.0) tCand = t2;
    if(tCand<0.0){ tSeg = -1.0; return false; }

    tSeg = tCand;
    hitP = xPrev + d * tCand;
    hitN = normalize(hitP - C);
    return true;
}

// Intersección rayo-plano y=const para segmento (xPrev -> xNow)
bool plane_cross_segment(vec3 xPrev, vec3 xNow, float y0, out float tSeg){
    float y1 = xPrev.y - y0;
    float y2 = xNow.y  - y0;
    if(y1==y2){ tSeg = -1.0; return false; }
    if( (y1>0.0 && y2<0.0) || (y1<0.0 && y2>0.0) ){
        tSeg = clamp(y1 / (y1 - y2), 0.0, 1.0);
        return true;
    }
    tSeg = -1.0;
    return false;
}

// “bloom” falso
vec3 faux_bloom(vec3 c, vec2 uv01, float amt){
    if(amt<=0.0) return c;
    float l = max(max(c.r,c.g),c.b);
    float h = smoothstep(0.6, 1.0, l);
    float r = length(uv01 - 0.5);
    float halo = smoothstep(0.9, 0.2, r);
    vec3 glow = c * h * halo * (0.25 + 0.75*amt);
    glow *= vec3(1.1, 1.05, 0.9);
    return c + glow;
}
vec3 edge_tint(vec3 c, vec2 uv01, float k){
    if(k<=0.0) return c;
    float r = length(uv01 - 0.5);
    float f = smoothstep(0.25, 0.8, r);
    vec3 tint = vec3(1.0 + 0.35*f, 1.0, 1.0 + 0.55*f);
    return c * mix(vec3(1.0), tint, 0.25*k);
}
vec3 film_dust(vec3 c, vec2 uv01, float t, float k){
    if(k<=0.0) return c;
    float lines = step(0.995, fract(uv01.x*200.0 + sin(t*1.7)*5.0));
    float specks = step(0.998, hash12(uv01*iResolution*0.75 + vec2(t*13.0, t*17.0)));
    float amt = k * 0.25;
    c += vec3(1.0) * (lines*0.12 + specks*0.08) * amt;
    return c;
}

void main(){
    // Jitter vintage
    vec2 jUV = jitter_uv(uv, floor(uTime*24.0)/24.0, uJitter);

    // Rayo inicial
    vec2 p = (jUV*2.0 - 1.0);
    p.x *= iResolution.x/iResolution.y;

    vec3 ro = uCamPos;
    vec3 fwd = normalize(uCamFwd);
    vec3 right = normalize(uCamRight);
    vec3 up    = normalize(uCamUp);

    vec3 rd0 = normalize(fwd + p.x * tan(uFov*0.5) * right + p.y * tan(uFov*0.5) * up);

    float M = uM;
    float aSpin = clamp(uA, 0.0, 0.99);
    float rho_h = M * 0.5; // horizonte isotrópico aprox

    vec3 x = ro;
    vec3 v = rd0;
    int   STEPS = int(uSteps);
    float h     = uH;

    bool hitBH    = false;
    bool hitPl    = false;
    bool hitDisk  = false;

    vec3 plCol    = vec3(0.0);
    vec3 diskCol  = vec3(0.0);

    float t_total = 0.0;

    for(int i=0;i<5000;i++){
        if(i>=STEPS) break;

        float rho = length(x);
        if(rho <= rho_h*1.001){
            hitBH = true;
            break;
        }

        // Aceleración óptica (Schwarzschild)
        vec3 gln = grad_logn(x, M);
        vec3 a_s = gln - dot(v, gln)*v;

        // Arrastre tipo Kerr
        float r_areal = rho_to_areal(rho, M);
        float omega = 2.0 * aSpin * M / max(r_areal*r_areal*r_areal, 1e-6);
        vec3 ephi = phi_hat(x);
        vec3 v_perp = ephi - dot(ephi, v)*v;
        vec3 a_fd = omega * v_perp;

        // RK2
        vec3 a1 = a_s + a_fd;
        vec3 x_mid = x + v * (0.5*h);
        vec3 v_mid = normalize(v + a1 * (0.5*h));

        vec3 gln2 = grad_logn(x_mid, M);
        float rho_mid = length(x_mid);
        float r_areal_mid = rho_to_areal(rho_mid, M);
        float omega_mid = 2.0 * aSpin * M / max(r_areal_mid*r_areal_mid*r_areal_mid, 1e-6);
        vec3 ephi_mid = phi_hat(x_mid);
        vec3 v_perp_mid = ephi_mid - dot(ephi_mid, v_mid)*v_mid;
        vec3 a_fd_mid = omega_mid * v_perp_mid;

        vec3 a2 = (gln2 - dot(v_mid, gln2)*v_mid) + a_fd_mid;

        // Avanzar
        vec3 x_prev = x;
        x += v_mid * h;
        v  = normalize(v + a2 * h);
        t_total += h;

        // --- Disco (plano y=0, aro entre radios areales) ---
        if(!hitDisk){
            float tP;
            if( plane_cross_segment(x_prev, x, 0.0, tP) ){
                vec3 xp = mix(x_prev, x, tP);
                float rP = length(xp.xz);
                float rA = rho_to_areal(rP, M);

                float inner = uDiskInnerR;
                float outer = uDiskOuterR;

                float w_in  = 1.0 - smoothstep(inner*0.9, inner, rA);
                float w_out = 1.0 - smoothstep(outer, outer*1.1, rA);
                float inside = clamp(w_in * w_out, 0.0, 1.0);

                if(inside > 0.0){
                    float phi = atan(xp.z, xp.x);
                    float stripe = 0.65 + 0.35*cos(phi*7.0);
                    float rim = smoothstep(inner, inner*1.1, rA);

                    vec3 base = mix(uDiskBaseCol, uDiskHotCol, stripe*rim);
                    base *= (0.7 + 0.5*rim);
                    diskCol = base * uDiskIntensity;
                    hitDisk = true;
                }
            }
        }

        // --- Planeta (esfera) ---
        if(!hitPl){
            float tS; vec3 P; vec3 N;
            if( sphere_cross_segment(x_prev, x, uPlPos, uPlRadius, tS, P, N) ){
                vec3 L = normalize(uSunDir);
                vec3 V = normalize(-v);  
                float NdotL = max(dot(N, L), 0.0);
                float ambient = uPlAmbient;
                float diff = NdotL;
                vec3 H = normalize(L + V);
                float spec = pow(max(dot(N, H), 0.0), uPlShininess) * uPlSpecular;
                float rim = pow(1.0 - max(dot(N, V), 0.0), 2.0) * uRimStrength;

                float lat = N.y;
                vec3 bandCol = mix(uPlAlbedo, uPlAlbedo*vec3(1.1,0.9,0.9), 0.4 + 0.4*lat);
                vec3 base = bandCol * (ambient + diff) + vec3(spec);
                base += uRimColor * rim;
                plCol = base;
                hitPl = true;
                break;
            }
        }

        if(t_total>250.0) break;
        if(length(x)>250.0) break;
    }

    // Fondo oscuro
    vec3 col = vec3(0.006, 0.005, 0.007);

    // Composición
    if(hitDisk) col = diskCol;
    if(hitPl)   col = plCol;
    if(hitBH)   col = vec3(0.0);

    // viñeta leve
    vec2 juv = jitter_uv(uv, uTime, uJitter);
    float d2 = dot(juv-0.5, juv-0.5);
    float vig = smoothstep(0.30, 0.78, 1.0 - d2);
    col *= vig;

    // exposición
    col = vec3(1.0) - exp(-col * uExposure);

    // “bloom” falso (suave)
    col = faux_bloom(col, uv, 0.55);

    // grading (lift/gamma/gain)
    col = max(col + uLift, vec3(0.0));
    col = pow(col, vec3(1.0 / max(uGamma, 0.05)));
    col *= uGain;

    // look vintage
    float flk = 1.0 + uFlicker * (sin(uTime*63.0)*0.02 + sin(uTime*117.0)*0.015);
    col = apply_sepia(col, uSepia);
    col = apply_scanlines(col, uv, uScan);
    col = edge_tint(col, uv, uEdgeTint);
    col = apply_grain(col, uv, uTime, uGrain);
    col = film_dust(col, uv, uTime, uDust);
    col *= flk;

    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""

QUAD = np.array([
    -1.0, -1.0,
     1.0, -1.0,
    -1.0,  1.0,
    -1.0,  1.0,
     1.0, -1.0,
     1.0,  1.0,
], dtype=np.float32)

# ================== Cámara y utilidades ==================
def clamp(x, a, b):
    return a if x < a else (b if x > b else x)

def yaw_pitch_to_basis(yaw, pitch):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    fwd = np.array([cy*cp, sp, -sy*cp], dtype=np.float32); fwd /= np.linalg.norm(fwd)
    right = np.array([ sy, 0.0,  cy], dtype=np.float32);   right /= np.linalg.norm(right)
    up = np.cross(right, fwd);                              up /= np.linalg.norm(up)
    return fwd, right, up

def main():
    # ---- Config base (escala física) ----
    width, height = 1280, 720
    fov_deg = 90.0
    base_steps   = 800.0
    base_hstep   = 0.05
    base_M       = 1.0
    aSpin        = 0.6
    expos        = 1.45

    # Disco (aro compacto)
    base_disk_inner = 9 * base_M
    base_disk_outer = 10 * base_M
    disk_base  = (1.0, 0.45, 0.02)
    disk_hot   = (1.0, 0.08, 0.02)
    disk_int   = 2.1

    # Planeta
    base_pl_radius   = 2.2
    pl_albedo   = (0.20, 0.45, 0.95)
    sun_dir     = (0.6, 0.35, 0.72)
    pl_specular = 0.28
    pl_shiny    = 72.0
    pl_ambient  = 0.10
    rim_color   = (0.25, 0.55, 1.0)
    rim_str     = 0.38

    # Órbita (radio base)
    base_orbit_r     = 16.0 * base_M
    orbit_w          = 0.16
    orbit_phase      = 0.0

    # Vintage / Estilo
    sepia   = 0.35
    grain   = 0.40
    scan    = 0.28
    flicker = 0.15
    jitter  = 0.15
    dust    = 0.25
    edge_t  = 0.45

    # Grading
    lift    = (0.02, 0.01, 0.0)
    gamma   = 1.05
    gain    = (1.05, 1.03, 0.98)

    # Cámara (vista base “bonita” de cerca)
    cam_pos = np.array([0.0, 2.4, 18.0], dtype=np.float32)
    yaw, pitch = 0.0, 0.03
    move_speed = 8.0
    mouse_sens_move, mouse_sens_rot = 0.02, 0.003

    # Distancia de referencia para el escalado
    base_cam_dist = float(np.linalg.norm(cam_pos))

    # ---- GLFW/GL ----
    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    glfw.window_hint(glfw.SAMPLES, 4)
    win = glfw.create_window(width, height, "Black Hole Kerr", None, None)
    glfw.make_context_current(win)
    ctx = moderngl.create_context()

    try:
        info = ctx.info
        print("[OpenGL] VENDOR :", info.get("GL_VENDOR"))
        print("[OpenGL] RENDERER:", info.get("GL_RENDERER"))
        print("[OpenGL] VERSION :", info.get("GL_VERSION"))
    except Exception:
        pass

    prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
    vbo = ctx.buffer(QUAD.tobytes())
    vao = ctx.simple_vertex_array(prog, vbo, "in_pos")

    # Mouse state
    last_x, last_y = glfw.get_cursor_pos(win)
    left_drag = right_drag = middle_drag = False
    scroll_acc = 0.0

    def scroll_cb(_w, xo, yo):
        nonlocal scroll_acc
        scroll_acc += yo
    glfw.set_scroll_callback(win, scroll_cb)

    glfw.set_input_mode(win, glfw.CURSOR, glfw.CURSOR_NORMAL)
    glfw.set_input_mode(win, glfw.STICKY_MOUSE_BUTTONS, glfw.TRUE)

    start = time.time()
    last = start
    while not glfw.window_should_close(win):
        now = time.time()
        dt = max(1e-6, now - last)
        last = now
        tsec = now - start

        glfw.poll_events()

        # Botones del mouse
        l_state = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT)
        r_state = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT)
        m_state = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_MIDDLE)

        if l_state == glfw.PRESS and not left_drag:
            left_drag = True
            glfw.set_input_mode(win, glfw.CURSOR, glfw.CURSOR_DISABLED)
            last_x, last_y = glfw.get_cursor_pos(win)
        if r_state == glfw.PRESS and not right_drag:
            right_drag = True
            glfw.set_input_mode(win, glfw.CURSOR, glfw.CURSOR_DISABLED)
            last_x, last_y = glfw.get_cursor_pos(win)
        if m_state == glfw.PRESS and not middle_drag:
            middle_drag = True
            glfw.set_input_mode(win, glfw.CURSOR, glfw.CURSOR_DISABLED)
            last_x, last_y = glfw.get_cursor_pos(win)

        if l_state == glfw.RELEASE and left_drag:
            left_drag = False
            if not (right_drag or middle_drag):
                glfw.set_input_mode(win, glfw.CURSOR, glfw.CURSOR_NORMAL)
        if r_state == glfw.RELEASE and right_drag:
            right_drag = False
            if not (left_drag or middle_drag):
                glfw.set_input_mode(win, glfw.CURSOR, glfw.CURSOR_NORMAL)
        if m_state == glfw.RELEASE and middle_drag:
            middle_drag = False
            if not (left_drag or right_drag):
                glfw.set_input_mode(win, glfw.CURSOR, glfw.CURSOR_NORMAL)

        # Cursor
        xpos, ypos = glfw.get_cursor_pos(win)
        dx = xpos - last_x
        dy = ypos - last_y
        last_x, last_y = xpos, ypos

        # Basis
        fwd, right, up = yaw_pitch_to_basis(yaw, pitch)

        # Rotar cámara (RMB)
        if right_drag:
            yaw   += dx * mouse_sens_rot
            pitch -= dy * mouse_sens_rot
            pitch = clamp(pitch, -1.2, 1.2)
            fwd, right, up = yaw_pitch_to_basis(yaw, pitch)

        # Mover en plano (LMB)
        if left_drag:
            cam_pos += (right * (dx * mouse_sens_move) + fwd * (-dy * mouse_sens_move)) * move_speed * dt

        # Subir/bajar (MMB)
        if middle_drag:
            cam_pos += up * (-dy * mouse_sens_move) * move_speed * dt

        # --------- ZOOM con rueda (DOLLY) ----------
        if abs(scroll_acc) > 0.0:
            zoom_step = 3.0          # ajusta si quieres más/menos agresivo
            cam_pos += fwd * (-scroll_acc * zoom_step)
            scroll_acc = 0.0
        # -------------------------------------------

        # ---- Escala dependiente de la distancia ----
        cam_dist = float(np.linalg.norm(cam_pos))
        scale = max(cam_dist / base_cam_dist, 1.0)

        M       = base_M * scale
        steps   = base_steps
        hstep   = base_hstep * scale

        disk_inner = base_disk_inner * scale
        disk_outer = base_disk_outer * scale
        orbit_r    = base_orbit_r * scale
        pl_radius  = base_pl_radius * scale

        # Órbita del planeta (XZ)
        ang = orbit_phase + orbit_w * tsec
        pl_pos = np.array([orbit_r*math.cos(ang), 0.0, orbit_r*math.sin(ang)], dtype=np.float32)

        # Render
        W, H = glfw.get_framebuffer_size(win)
        ctx.viewport = (0, 0, W, H)
        ctx.clear(0.01, 0.01, 0.015, 1.0)

        # Uniformes principales
        prog["iResolution"].value = (float(W), float(H))
        prog["uTime"].value       = float(tsec)
        prog["uFov"].value        = float(math.radians(fov_deg))
        prog["uSteps"].value      = float(steps)
        prog["uH"].value          = float(hstep)
        prog["uM"].value          = float(M)
        prog["uA"].value          = float(aSpin)
        prog["uExposure"].value   = float(expos)

        # Disco
        prog["uDiskInnerR"].value    = float(disk_inner)
        prog["uDiskOuterR"].value    = float(disk_outer)
        prog["uDiskBaseCol"].value   = tuple(map(float, disk_base))
        prog["uDiskHotCol"].value    = tuple(map(float, disk_hot))
        prog["uDiskIntensity"].value = float(disk_int)

        # Vintage / estilo
        prog["uSepia"].value    = float(sepia)
        prog["uGrain"].value    = float(grain)
        prog["uScan"].value     = float(scan)
        prog["uFlicker"].value  = float(flicker)
        prog["uJitter"].value   = float(jitter)
        prog["uDust"].value     = float(dust)
        prog["uEdgeTint"].value = float(edge_t)

        # Grading
        prog["uLift"].value   = tuple(map(float, lift))
        prog["uGamma"].value  = float(gamma)
        prog["uGain"].value   = tuple(map(float, gain))

        # Cámara
        prog["uCamPos"].value   = tuple(map(float, cam_pos.tolist()))
        prog["uCamFwd"].value   = tuple(map(float, fwd.tolist()))
        prog["uCamRight"].value = tuple(map(float, right.tolist()))
        prog["uCamUp"].value    = tuple(map(float, up.tolist()))

        # Planeta + luz
        sdir = np.array(sun_dir, dtype=np.float32)
        sdir /= np.linalg.norm(sdir) + 1e-9
        prog["uPlPos"].value      = tuple(map(float, pl_pos.tolist()))
        prog["uPlRadius"].value   = float(pl_radius)
        prog["uPlAlbedo"].value   = tuple(map(float, pl_albedo))
        prog["uSunDir"].value     = (float(sdir[0]), float(sdir[1]), float(sdir[2]))
        prog["uPlSpecular"].value = float(pl_specular)
        prog["uPlShininess"].value= float(pl_shiny)
        prog["uPlAmbient"].value  = float(pl_ambient)
        prog["uRimColor"].value   = tuple(map(float, rim_color))
        prog["uRimStrength"].value= float(rim_str)

        vao.render()
        glfw.swap_buffers(win)

    glfw.terminate()

if __name__ == "__main__":
    main()
