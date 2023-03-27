"""
author: aaron pache (email: lpxap7@nottingham.ac.uk)
"""
"""
All code typed by me. 

Oklab converts the original implementation into python: https://bottosson.github.io/posts/colorpicker/

Other colorspaces use: http://brucelindbloom.com/
and is also an excellent resource on color!

Utility functions occur between lines 27 - 827. These define the colorspaces,
color conversions, color distance functions, etc.

Below line 827 are key generator functions. The most used (and updated) are 
below line 890. These use the oklab colorspace which attempts to be 
perceptually uniform. 

You can generate a wheel of hues, a set of color gradients or both.

At the bottom (line 965) is a set of example use cases.
"""
import matplotlib.pyplot as plt
import numpy as np


def normalise(x, n, mn, mx):
    step = (mx-mn)/n
    return mn+(x*step)

def c_lerp(colour1, colour2, n):
    x1, y1, z1 = colour1
    x2, y2, z2 = colour2
    
    dx = (x2 - x1)/n
    dy = (y2 - y1)/n
    dz = (z2 - z1)/n
    
    lerp = np.zeros((n, 3))
    for i in range(0, n):
        lerp[i] = x1+dx*i, y1+dy*i, z1+dz*i
    return lerp
        
def rgb_to_hsv(rgb):
    r, g, b, = rgb
    cmax = np.max(rgb)
    cmin = np.min(rgb)
    
    delta = cmax - cmin
    
    if cmax == cmin:
        h = 0.0
    else:
        if cmax == r:
            h = ((g - b) / delta) % 6
        elif cmax == g:
            h = (b - r) / delta + 2
        else:
            h = (r - g) / delta + 4
            
    h = h * 60
        
    if cmax != 0:
        s = delta / cmax
    else:
        s = 0
        
    v = cmax
    
    return np.array([h, s, v])

def hsv_to_rgb(hsv):
    h, s, v = hsv
    c = v * s
    
    x = c * (1 - abs(((h / 60) % 2) - 1))
    
    m = v - c
    
    
    if h >= 360:
        h = h - 360
    
    if 0 <= h < 60:
        rgb = np.array([c, x, 0])
    elif 60 <= h < 120:
        rgb = np.array([x, c, 0])
    elif 120 <= h < 180:
        rgb = np.array([0, c, x])
    elif 180 <= h < 240:
        rgb = np.array([0, x, c])
    elif 240 <= h < 300:
        rgb = np.array([x, 0, c]) 
    else:
        rgb = np.array([c, 0, x])
    
    r, g, b = rgb + m
    
    return np.array([r, g, b]) 

def hsv_to_hsl(hsv):
    h, s, v = hsv
    hl = h
    
    l = v*(1 - s/2)
    
    if l != 0 and l != 1:
        sl = (v - l)/min(l, 1-l)
    else:
        sl = 0
        
    return np.array([hl, sl, l])

def hsl_to_hsv(hsl):
    h, s, l = hsl
    hv = h
    
    v = l + s * min(l, 1-l)
    
    if v != 0:
        sv = 2*(1 - l/v)
    else:
        sv = 0
        
    return np.array([hv, sv, v])

def gamma(x):
    return np.where(x >= 0.0031308, 1.055*np.power(x*(x>0), 1/2.4)-0.055, 12.92*x)
    
def gamma_inv(x):
    return np.where(x >= 0.04045, np.power((x+0.055)/1.055, 2.4), x/12.92)

def lin_to_rgb(r, g, b):
    rgb = np.array([r, g, b])
    rgb = np.where(rgb >= 0.04045, np.power((rgb+0.055)/1.055, 2.4), rgb/12.92)
    return rgb

def rgb_to_lin(r, g, b):
    rgb = np.array([r, g, b])
    rgb = np.where(rgb < 0.0031308, 12.92*rgb, 1.055*np.power(rgb, 1/2.4) - 0.055)
    return rgb



"""
# oklab code: https://bottosson.github.io/posts/colorpicker/
# intro to oklab: https://bottosson.github.io/posts/oklab/
"""


def rgb_to_oklab(rgb):
    rgb = gamma_inv(rgb)
    #rgb = lin_to_rgb(r, g, b)
    M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    
    lms = np.dot(M1, rgb)
    
    lms = np.power(lms, 1/3)
    
    M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    
    return np.dot(M2, lms)

def oklab_to_rgb(lab):
    #lab = np.array([l, a, b])
    #print(lab)
    M1 = np.array([[1.0000000000, 0.3963377774, 0.2158037573],
                  [1.0000000000, -0.1055613458, -0.0638541728],
                  [1.0000000000, -0.0894841775, -1.2914855480]])
    lms = np.dot(M1, lab)
    
    lms = np.power(lms, 3)
    
    M2 = np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                  [-1.2684380046, 2.6097574011, -0.3413193965],
                  [-0.0041960863, -0.7034186147, 1.7076147010]])
    rgb = np.dot(M2, lms)
    #r, g, b = rgb
    return gamma(rgb)

def xyz_to_oklab(xyz):
    print(xyz)
    M1 = np.array([[0.8189330101, 0.3618667424, -0.1288597137],
                  [0.0329845436, 0.9293118715, 0.0361456387],
                  [0.0482003018, 0.2643662691, 0.6338517070]])
    M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    lms = np.dot(M1, xyz)
    #print(lms)
    
    lms_ = np.power(lms, 1/3)
    
    return np.round(np.dot(M2, lms_), 15)

def oklab_to_xyz(lab):
    M1 = np.array([[0.8189330101, 0.3618667424, -0.1288597137],
                  [0.0329845436, 0.9293118715, 0.0361456387],
                  [0.0482003018, 0.2643662691, 0.6338517070]])
    M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    M1_inv = np.linalg.inv(M1)
    M2_inv = np.linalg.inv(M2)    
    
    lms = np.dot(M2_inv, lab)
    
    lms_ = np.power(lms, 3)
    
    return np.round(np.dot(M1_inv, lms_), 15)


def lab_to_lch(lab):
    l, a, b = lab
    c = np.sqrt(a*a + b*b)
    h = np.rad2deg(np.arctan2(b, a))
    if h < 0:
        h += 360
    return np.array([l, c, h])

def lch_to_lab(lch):
    l, c, h = lch
    a = c*np.cos(np.deg2rad(h))
    b = c*np.sin(np.deg2rad(h))
    return np.array([l, a, b])

def toe(l):
    k1 = 0.206
    k2 = 0.03
    k3 = (1+k1)/(1+k2)
    
    lr = (k3*l-k1+np.sqrt(np.square(k3*l-k1)+4*k2*k3*l))/2
         
    return lr

def toe_inv(lr):
    k1 = 0.206
    k2 = 0.03
    k3 = (1+k1)/(1+k2)
    
    l = (lr*(lr+k1))/(k3*(lr+k2))

    return l

def compute_max_saturation(a, b):
    if -1.88170328 * a - 0.80936493 * b > 1:
        k0 = +1.19086277
        k1 = +1.76576728
        k2 = +0.59662641
        k3 = +0.75515197
        k4 = +0.56771245
        wl = +4.0767416621
        wm = -3.3077115913 
        ws = +0.2309699292
    elif 1.81444104 * a - 1.19445276 * b > 1:
        k0 = +0.73956515
        k1 = -0.45954404
        k2 = +0.08285427
        k3 = +0.12541070
        k4 = +0.14503204
        wl = -1.2684380046 
        wm = +2.6097574011 
        ws = -0.3413193965
    else:
        k0 = +1.35733652
        k1 = -0.00915799
        k2 = -1.15130210
        k3 = -0.50559606
        k4 = +0.00692167
        wl = -0.0041960863
        wm = -0.7034186147
        ws = +1.7076147010
        
    S = k0 + k1 * a + k2 * b + k3 * a * a + k4 * a * b
    
    k_l = +0.3963377774 * a + 0.2158037573 * b
    k_m = -0.1055613458 * a - 0.0638541728 * b
    k_s = -0.0894841775 * a - 1.2914855480 * b
    
    l_ = 1 + S * k_l
    m_ = 1 + S * k_m
    s_ = 1 + S * k_s

    l = l_ * l_ * l_
    m = m_ * m_ * m_
    s = s_ * s_ * s_

    l_dS = 3 * k_l * l_ * l_
    m_dS = 3 * k_m * m_ * m_
    s_dS = 3 * k_s * s_ * s_

    l_dS2 = 6 * k_l * k_l * l_
    m_dS2 = 6 * k_m * k_m * m_
    s_dS2 = 6 * k_s * k_s * s_

    f  = wl * l     + wm * m     + ws * s
    f1 = wl * l_dS  + wm * m_dS  + ws * s_dS
    f2 = wl * l_dS2 + wm * m_dS2 + ws * s_dS2

    S = S - f * f1 / (f1*f1 - 0.5 * f * f2)
    
    return S

def find_cusp(a, b):
    
    S_cusp = compute_max_saturation(a, b)
    
    rgb_at_max = gamma_inv(oklab_to_rgb(np.array([1, S_cusp*a, S_cusp*b])))
    
    L_cusp = np.power(1/(np.max(rgb_at_max)), 1/3)
    C_cusp = L_cusp*S_cusp
    
    return (L_cusp, C_cusp)

def find_gamut_intersection(a, b, L1, C1, L0, n_iter=1, cusp=None):
    
    if cusp is None:
        cusp = find_cusp(a, b)
    
    #print(cusp)
    #print(cusp[0])
    #print(L1)
    #print(L0)
    #print(C1)
    #print(L0)

    if ((L1 - L0)*cusp[1] - (cusp[0] - L0) * C1) <= 0:
        t = cusp[1] * L0 / (C1 * cusp[0] + cusp[1] * (L0 - L1))
    else:
        t = cusp[1] * (L0 - 1) / (C1 * (cusp[0] - 1) + cusp[1] * (L0 - L1));
        dL = L1 - L0
        dC = C1 
        k_l = +0.3963377774 * a + 0.2158037573 * b
        k_m = -0.1055613458 * a - 0.0638541728 * b
        k_s = -0.0894841775 * a - 1.2914855480 * b
        
        l_dt = dL + dC * k_l
        m_dt = dL + dC * k_m
        s_dt = dL + dC * k_s
        
        for i in range(n_iter):
            
            L = L0 * (1 - t) + t * L1
            C = t * C1
                        
            l_ = L + C * k_l
            m_ = L + C * k_m
            s_ = L + C * k_s
            
            l = l_ * l_ * l_
            m = m_ * m_ * m_
            s = s_ * s_ * s_
            
            ldt = 3 * l_dt * l_ * l_
            mdt = 3 * m_dt * m_ * m_
            sdt = 3 * s_dt * s_ * s_
            
            ldt2 = 6 * l_dt * l_dt * l_
            mdt2 = 6 * m_dt * m_dt * m_
            sdt2 = 6 * s_dt * s_dt * s_
            
            r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s - 1
            r1 = 4.0767416621 * ldt - 3.3077115913 * mdt + 0.2309699292 * sdt
            r2 = 4.0767416621 * ldt2 - 3.3077115913 * mdt2 + 0.2309699292 * sdt2
            
            u_r = r1 / (r1 * r1 - 0.5 * r * r2)
            t_r = -r * u_r
            
            g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s - 1
            g1 = -1.2684380046 * ldt + 2.6097574011 * mdt - 0.3413193965 * sdt
            g2 = -1.2684380046 * ldt2 + 2.6097574011 * mdt2 - 0.3413193965 * sdt2
            
            u_g = g1 / (g1 * g1 - 0.5 * g * g2)
            t_g = -g * u_g
            
            b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s - 1
            b1 = -0.0041960863 * ldt - 0.7034186147 * mdt + 1.7076147010 * sdt
            b2 = -0.0041960863 * ldt2 - 0.7034186147 * mdt2 + 1.7076147010  * sdt2
            
            u_b = b1 / (b1 * b1 - 0.5 * b * b2)
            t_b = -b * u_b
            
            if u_r < 0:
                t_r = 10e5
            if u_g < 0:
                t_g = 10e5
            if u_b < 0:
                t_b = 10e5  
            t += np.min(np.array([t_r, t_g, t_b]))
    return t

def get_ST_max(a_, b_, cusp=None):
    
    if cusp is None:
        cusp = find_cusp(a_, b_)
    
    L, C = cusp
    
    return (C/L, C/(1-L))

def get_ST_mid(a_, b_):
    
    S = 0.11516993 + 1/(
        + 7.44778970 + 4.15901240*b_
        + a_*(- 2.19557347 + 1.75198401*b_
        + a_*(- 2.13704948 -10.02301043*b_ 
        + a_*(- 4.24894561 + 5.38770819*b_ + 4.69891013*a_
        )))
    )
    
    T = 0.11239642 + 1/(
        + 1.61320320 - 0.68124379*b_
        + a_*(+ 0.40370612 + 0.90148123*b_
        + a_*(- 0.27087943 + 0.61223990*b_ 
        + a_*(+ 0.00299215 - 0.45399568*b_ - 0.14661872*a_
        )))
    )

    return (S, T)

def get_Cs(L, a_, b_):
    
    cusp = find_cusp(a_, b_)
    
    C_max = find_gamut_intersection(a_, b_, L, 1, L, cusp=cusp)
    ST_max = get_ST_max(a_, b_, cusp)
    
    S_mid = 0.11516993 + 1/(
            + 7.44778970 + 4.15901240*b_
            + a_*(- 2.19557347 + 1.75198401*b_
            + a_*(- 2.13704948 -10.02301043*b_ 
            + a_*(- 4.24894561 + 5.38770819*b_ + 4.69891013*a_
            )))
        ) 
    
    T_mid = 0.11239642 + 1/(
        + 1.61320320 - 0.68124379*b_
        + a_*(+ 0.40370612 + 0.90148123*b_
        + a_*(- 0.27087943 + 0.61223990*b_ 
        + a_*(+ 0.00299215 - 0.45399568*b_ - 0.14661872*a_
        )))
    )
    
    #print(L)
    #print(ST_max)
    #print(C_max)
    
    k = C_max/np.min([(L*ST_max[0]), (1-L)*ST_max[1]])
    
    C_a = L*S_mid
    C_b = (1-L)*T_mid
    
    C_mid = 0.9*k*np.sqrt(np.sqrt(1/(1/(C_a*C_a*C_a*C_a) + 1/(C_b*C_b*C_b*C_b))))
    
    C_a = L*0.4
    C_b = (1-L)*0.8

    C_0 = np.sqrt(1/1/(C_a*C_a) + 1/(C_b*C_b))

    return (C_0, C_mid, C_max)

def okhsl_to_rgb(hsl, clip=True):
    h, s, l = hsl    
    if l == 1:
        return np.array([1, 1, 1])
    if l == 0:
        return np.array([0, 0, 0])
    
    a_ = np.cos(np.deg2rad(h))
    b_ = np.sin(np.deg2rad(h))
    L = toe_inv(l)
    
    C_0, C_mid, C_max = get_Cs(L, a_, b_)
    
    if s < 0.8:
        t = 1.25*s
        k_0 = 0
        k_1 = 0.8*C_0
        k_2 = (1-k_1/C_mid)
    else:
        t = 5*(s-0.8)
        k_0 = C_mid
        k_1 = 0.2*C_mid*C_mid*1.25*1.25/C_0
        k_2 = (1- (k_1)/(C_max - C_mid))
        
    C = k_0 + t*k_1/(1-k_2*t)
    
    rgb = oklab_to_rgb(np.array([L, C*a_, C*b_]))
    if clip:    
        return np.clip(rgb, 0,1)
    else:
        return rgb

def rgb_to_okhsl(rgb):
    l, a, b = rgb_to_oklab(rgb)
    C = np.sqrt(a*a+b*b)
    a_ = a/C
    b_ = b/C
    
    L = l
    h = (0.5 + 0.5 * np.arctan2(-b, -a)/np.pi)*360
    
    C_0, C_mid, C_max = get_Cs(L, a_, b_)
    
    mid = 0.8
    mid_inv = 1.25
    
    if C < C_mid:
        k_1 = mid * C_0
        k_2 = (1 - k_1/C_mid)
        
        t = C/(k_1 + k_2 * C)
        s = t * mid
    else:
        k_0 = C_mid
        k_1 = (1.0 - mid) * (C_mid * C_mid * C_mid * mid_inv * mid_inv)/C_0
        k_2 = (1.0 - k_1) / (C_max - C_mid)
        t = (C - k_0) / (k_1 + k_2 * (C - k_0))
        s = mid + (1.0 - mid) * t
        
    l = toe(L)
    
    return np.array([h, s, l])

def okhsv_to_rgb(hsv, clip=True):
    h, s, v = hsv
    a_ = np.cos(np.deg2rad(h))
    b_ = np.sin(np.deg2rad(h))
    
    ST_max = get_ST_max(a_, b_)
    S_max = ST_max[0]
    S_0 = 0.5
    T = ST_max[1]
    k = 1 - S_0/S_max
    
    L_v = 1 - s*S_0/(S_0+T - T*k*s)
    C_v = s*T*S_0/(S_0+T-T*k*s)
    
    L = v*L_v
    C = v*C_v
    
    L_vt = toe_inv(L_v)
    C_vt = C_v* L_vt/L_v
    
    L_new = toe_inv(L)
    C = C * L_new/L
    L = L_new
    
    rgb_scale = gamma_inv(oklab_to_rgb(np.array([L_vt, a_*C_vt, b_*C_vt])))
    rgb_scale = np.append(rgb_scale, 0)
    scale_L = np.power(1/np.max(rgb_scale), 1/3)
    
    L = L*scale_L
    C = C*scale_L
    if clip:
        return np.clip(oklab_to_rgb(np.array([L, C*a_, C*b_])), 0, 1)
    else:
        return oklab_to_rgb(np.array([L, C*a_, C*b_]))
    
def rgb_to_okhsv(rgb):
    l, a, b = rgb_to_oklab(rgb)
    
    C = np.sqrt(a*a + b*b)
    a_ = a/C
    b_ = b/C
    
    L = l
    
    h = (0.5 + 0.5*np.arctan2(-b, -a) / np.pi)*360
    
    cusp = find_cusp(a_, b_)
    
    ST_max = get_ST_max(a_, b_, cusp=cusp)
    S_max = ST_max[0]
    
    S_0 = 0.5
    
    T = ST_max[1]
    k = 1 - S_0/S_max
    
    t = T/(C+L*T)
    L_v = t*L
    C_v = t*C
    
    L_vt = toe_inv(L_v)
    C_vt = C_v * L_vt/L_v
    
    rgb_scale = gamma_inv(oklab_to_rgb(np.array([L_vt, a_*C_vt, b_*C_vt])))
    rgb_scale = np.append(rgb_scale, 0)
    
    scale_L = np.power(1/(np.max(rgb_scale)), 1/3)
    
    L = L/scale_L
    C = C/scale_L
    
    C = C * toe(L)/L
    L = toe(L)
    
    v = L/L_v
    s = (S_0+T) * C_v/((T*S_0) + T*k*C_v)
    
    return np.array([h, s, v])

"""
deltaE can be used for determining rough color distance in the CIELAB colorspace (hence 'lab1', 'lab2')
"""


def deltaE(lab1, lab2):
    l1, a1, b1 = lab1
    l2, a2, b2 = lab2
    #print(l1, a1, b1)
    
    l_prime = (l1 + l2)/2.0
    
    c1 = np.sqrt(a1*a1 + b1*b1)
    c2 = np.sqrt(a2*a2 + b2*b2)
    
    c_bar = (c1 + c2)/2.0
    
    g = 0.5*(1.0-np.sqrt(np.power(c_bar, 7.0)/(np.power(c_bar, 7.0) + 25.0**7.0)))
    
    a1_prime = a1*(1.0 + g)
    a2_prime = a2*(1.0 + g)
    
    c1_prime = np.sqrt(a1_prime*a1_prime + b1*b1)
    c2_prime = np.sqrt(a2_prime*a2_prime + b2*b2)
    
    c_bar_prime = (c1_prime + c2_prime)/2.0
    
    h1_prime = np.rad2deg(np.arctan2(b1, a1_prime))
    if h1_prime < 0.0:
        h1_prime += 360.0
        
    h2_prime = np.rad2deg(np.arctan2(b2, a2_prime))
    if h2_prime < 0.0:
        h2_prime += 360.0
    
    if np.fabs(h1_prime - h2_prime) <= 180:
        H_bar_prime = (h1_prime + h2_prime)/2.0
    elif np.fabs(h1_prime - h2_prime) > 180.0 and (h1_prime + h2_prime) < 360:
        H_bar_prime = (h1_prime + h2_prime + 360.0)/2.0 
    elif np.fabs(h1_prime - h2_prime) > 180 and (h1_prime + h2_prime) >= 360:
        H_bar_prime = (h1_prime + h2_prime - 360)/2.0
    else:
        H_bar_prime = h1_prime + h2_prime
            
    t = (1.0 - 0.17*np.cos(np.deg2rad(H_bar_prime - 30.0)) + 
             0.24*np.cos(np.deg2rad(2.0*H_bar_prime)) + 
             0.32*np.cos(np.deg2rad(3.0*H_bar_prime+6.0)) - 
             0.20*np.cos(np.deg2rad(4.0*H_bar_prime - 63.0))
             )
    
    if np.abs(h2_prime - h1_prime) <= 180.0:    
        delta_h_prime = h2_prime - h1_prime 
    elif np.abs(h2_prime - h1_prime) > 180.0 and h2_prime <= h1_prime:
        delta_h_prime = h2_prime - h1_prime + 360.0
    else:
        delta_h_prime = h2_prime - h1_prime - 360.0
       
    delta_l_prime = l2 - l1
    delta_c_prime = c2_prime - c1_prime
    
    delta_H_prime = 2.0*np.sqrt(c1_prime*c2_prime)*np.sin(np.deg2rad(delta_h_prime/2.0))
    
    s_l = 1.0 + (0.015*np.square(l_prime - 50.0))/np.sqrt(20.0+np.square(l_prime-50.0))
    
    s_c = 1.0 + 0.045*c_bar_prime
    s_h = 1.0 + 0.015*c_bar_prime*t
    
    delta_theta = 30.0*np.exp(-np.square((H_bar_prime - 275.0)/25.0))
        
    r_c = 2.0*np.sqrt(np.power(c_bar_prime, 7.0)/(np.power(c_bar_prime, 7.0) + 25.0**7.0))
    r_t = -r_c*np.sin(np.deg2rad(2.0*delta_theta))
    
    k_l = 1.0
    k_c = 1.0
    k_h = 1.0
    
    delta_e = np.sqrt(np.square(delta_l_prime/(k_l*s_l)) +
                      np.square(delta_c_prime/(k_c*s_c)) + 
                      np.square(delta_H_prime/(k_h*s_h)) + 
                      r_t*(delta_c_prime/(k_c*s_c))*(delta_H_prime/(k_h*s_h))
                      )
    
    return delta_e

"""
If it is perceptually uniform then L1 should suffice.
It's quick and dirty but works surprisingly well.
"""
def deltaE_oklab(lab1, lab2):
    return np.linalg.norm(lab1 - lab2)

"""
Ensure colors meet the sRGB color of matplotlib
"""
def oklab_clip(rgb, alpha=2, clip=True):
    
    if (rgb < 1).all() and (rgb > 0).all():
        return rgb
    
    L, a, b = rgb_to_oklab(rgb)
    
    eps = 0.00001
    
    C = max(eps, np.sqrt(a*a + b*b))
    a_ = a/C
    b_ = b/C
    
    cusp = find_cusp(a_, b_)
    
    Ld = L - cusp[0]
    
    if Ld > 0:
        k = 2.0 * (1.0 - cusp[0])
    else:
        k = 2.0 * cusp[0]
    
    e1 = 0.5 * k + np.fabs(Ld) + alpha * C/k
    L0 = cusp[0] + 0.5 * (np.sign(Ld) * (e1 - np.sqrt(e1*e1 - 2.0*k*np.fabs(Ld))))
    
    t = find_gamut_intersection(a_, b_, L, C, L0, cusp=cusp)
    L_clipped = L0 * (1.0 - t) + t * L
    C_clipped = t * C
    
    if clip:
        return np.clip(oklab_to_rgb(np.array([L_clipped, C_clipped*a_, C_clipped*b_])), 0, 1)
    else:
        return oklab_to_rgb(np.array([L_clipped, C_clipped*a_, C_clipped*b_]))
    

def rgb_to_xyz(rgb):
    #rgb = np.array([r, g, b])
    rgb = gamma_inv(rgb)#np.where(rgb > 0.04045, np.power((rgb+0.055)/1.055, 2.4), rgb/12.92)
    
    M = np.array([[0.4124564, 0.3575761, 0.1804375], 
                  [0.2126729, 0.7151522, 0.0721750], 
                  [0.0193339, 0.1191920, 0.9503041]])
    
    xyz = np.dot(M, rgb)
    
    ref_white = np.array([0.95047  , 1.0000001, 1.08883  ])
    
    xyz = xyz/ref_white
    
    return np.round(xyz, 7)
    
def xyz_to_rgb(xyz):
    #xyz = np.array([x, y, z])
    ref_white = np.array([0.95047  , 1.0000001, 1.08883  ])
    xyz = xyz*ref_white
    
    inv_M = np.linalg.inv(np.array([[0.4124564, 0.3575761, 0.1804375], 
                                    [0.2126729, 0.7151522, 0.0721750], 
                                    [0.0193339, 0.1191920, 0.9503041]]))
    rgb = np.dot(inv_M, xyz)
    
    rgb = gamma(rgb)#np.where(rgb <= 0.0031308, 12.92*rgb, 1.055*np.power(rgb, 1/2.4) - 0.055)
    
    #rgb = rgb/ref_white
    
    return np.round(rgb, 7)

def xyz_to_cielab(xyz):
    
    x, y, z = xyz
    
    eps = 216/24389
    kappa = 24389/27

    
    fx, fy, fz = np.where(xyz > eps, np.cbrt(xyz), (kappa*xyz+16.0)/116.0)
    
    L = 116.0*fy-16.0
    a = 500.0*(fx-fy)
    b = 200.0*(fy-fz)
    
    return np.array([L, a, b])

def cielab_to_xyz(lab):
    l, a, b = lab
    eps = 216/24389
    kappa = 24389/27
    
    fy = (l + 16.0)/116.0
    fx = a/500.0 + fy
    fz = fy - b/200.0
    
    if np.power(fx, 3.0) > eps:
        x = np.power(fx, 3.0)
    else:
        x = (116.0*fx-16.0)/kappa
    
    if l > kappa*eps:
        y = np.power((l+16.0)/116.0, 3.0)
    else:
        y = l/kappa
    
    if np.power(fz, 3.0) > eps:
        z = np.power(fz, 3.0)
    else:
        z = (116.0*fz-16.0)/kappa
        
    return np.array([x, y, z])

def rgb_to_cielab(rgb):
    xyz = rgb_to_xyz(rgb)
    return xyz_to_cielab(xyz)

def cielab_to_rgb(lab, clip=True):
    xyz = cielab_to_xyz(lab)
    if clip:
        return np.clip(xyz_to_rgb(xyz), 0, 1)
    else:
        return xyz_to_rgb(xyz)

def oklab_mat(r, g, b):
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 *b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    return l, m, s

"""
Functions!
"""

## HSL (left behind since I just use oklab)
def gen_tones(n_colors, h=30, l=0.5):
    palette = np.zeros((n_colors, 3))
    for i in range(n_colors):
        h, s, v = hsl_to_hsv(h, normalise(i, n_colors, 0, 1), l)
        r, g, b = hsv_to_rgb(h, s, v)
        palette[i] = np.array([r, g, b])
    return palette

def gen_tints(n_colors, h=30, sl=1):
    palette = np.zeros((n_colors, 3))
    for i in range(n_colors):
        hsv = hsl_to_hsv(np.array([h, sl, normalise(i, n_colors, 0.5, 1)]))
        r, g, b = hsv_to_rgb(hsv)
        palette[i] = np.array([r, g, b])
    return palette

def gen_shades(n_colors, h=30, sl=1):
    palette = np.zeros((n_colors, 3))
    for i in range(n_colors):
        h, sv, v = hsl_to_hsv(np.array([h, sl, normalise(i, n_colors, 0.0, 0.5)]))
        r, g, b = hsv_to_rgb(h, sv, v)
        palette[i] = np.array([r, g, b])
    return palette

def gen_gradients(n_colors, h=30, sl=1, min_luminance=0, max_luminance=1):
    palette = np.zeros((n_colors, 3))
    for i in range(n_colors):
        hsv = hsl_to_hsv(np.array([h, sl, normalise(i, n_colors, min_luminance, max_luminance)]))
        r, g, b = hsv_to_rgb(hsv)
        palette[i] = np.array([r, g, b])
    return palette

def gen_wheel(n_colors, init_hue=30, s=1, l=0.5):
    palette = np.zeros((n_colors, 3))
    step = (360/n_colors)*(n_colors-1)
    for i in range(n_colors):
        hsv = hsl_to_hsv(np.array([i*step+init_hue, s, l]))
        r, g, b = hsv_to_rgb(hsv)
        palette[i] = np.array([r, g, b])
    return palette

def gen_color_set(n_colors, n_gradients, init_hue=30, s=1, l=0.5):
    palette = np.zeros((n_colors*n_gradients, 3))
    step = (360/n_colors)*(n_colors-1)
    #print(step)
    for i in range(n_colors):
        hsv = hsl_to_hsv(np.array([i*step+init_hue, s, l]))
        #print(h)
        r, g, b = hsv_to_rgb(hsv)
        #palette[i*n_gradients] = np.array([r, g, b])
        grads = gen_gradients(n_gradients, h=init_hue, sl=s, min_luminance=0.3, max_luminance=0.95)
        for j in range(n_gradients):
            palette[i*n_gradients+j] = grads[j]
            #print(i*n_gradients+j)
    return palette

"""
oklab
the oklab attempts to be perceptually uniform 
these generators attempt to make palettes with equi-distant colors in terms of perception
"""
# generate an RGB color with a hue and lightness from the oklab colorspace
def gen_ok_color(init_hue, l, s=1):
    return okhsl_to_rgb(np.array([init_hue, s, l]))

# generate a range of color gradients (increasing lightness) from the oklab colorspace
# good for plotting variation of a single variable/parameter
def gen_ok_gradients(n_colors, init_hue=30, s=1, min_l=0.45, max_l=0.95, shape=None, descending=True):
    palette = np.zeros((n_colors, 3))
    for i in range(n_colors):
        palette[i] = okhsl_to_rgb(np.array([init_hue, s, normalise(i, n_colors, min_l, max_l)]))
    if descending:
        palette = np.flip(palette, 0)
    if shape is not None:
        palette.reshape(shape)
    return palette

# generate a range of RGB colors (different hues) from the oklab colorspace
# good for single parameter, multiple conditions
def gen_ok_wheel(n_colors, init_hue=30, s=1, l=0.63, shape=None):
    palette = np.zeros((n_colors, 3))
    step = (360/n_colors)*(n_colors-1)
    for i in range(n_colors):
        palette[i] = okhsl_to_rgb(np.array([i*step+init_hue, s, l]))
    return palette

# generate a range of RGB colors and gradients from the oklab colorspace
# good for multiple conditions with varying parameters
# I almost exclusively use this function
def gen_ok_color_set(n_colors, n_gradients, init_hue=30, s=1, min_l=0.45, max_l=0.95, alpha=None, step=0, shape=None, descending=True):
    if alpha is None:
        palette = np.zeros((n_colors*n_gradients, 3))
    else:
        palette = np.zeros((n_colors*n_gradients, 4))
    if step == 0:
        step = (360/n_colors)*(n_colors-1)
    #print(step)
    for i in range(n_colors):
        grads = gen_ok_gradients(n_gradients, init_hue=i*step+init_hue, s=s, min_l=min_l, max_l=max_l, descending=descending)
        if alpha is None:
            for j in range(n_gradients):
                palette[i*n_gradients+j] = grads[j]
        else:
            if type(alpha) == float:
                for j in range(n_gradients):
                    palette[i*n_gradients+j] = np.append(grads[j], alpha)
            elif type(alpha) == list or type(alpha) == np.ndarray and len(alpha) == n_colors*n_gradients:
                for j in range(n_gradients):
                    palette[i*n_gradients+j] = np.append(grads[j], alpha[i*n_gradients+j])
            else:
                print('Error:')
            
            #print(i*n_gradients+j)
    if shape is not None:
        if alpha is None:
            palette = palette.reshape(shape+(3,))
        else:
            palette = palette.reshape(shape+(4,))
    return palette

# oklab transition from one color to another
# color arguments are from the rgb space: min=[R=0, G=0, B=0], max=[R=1, G=1, B=1]
def ok_transition(rgb1, rgb2, n):
    ok1 = rgb_to_oklab(rgb1)
    ok2 = rgb_to_oklab(rgb2)
    ok_colors = c_lerp(ok1, ok2, n=n)
    palette = np.zeros(ok_colors.shape)
    for i in range(len(ok_colors)):
        palette[i] = np.clip(oklab_to_rgb(ok_colors[i]), 0, 1)
    return palette

"""
    Uncomment below to see example use cases
"""

"""
# Example use cases:
# multi-condition
walk = np.arange(10)
bike = np.arange(10)*2
car = np.arange(10)*3
distance = np.asarray((walk, bike, car))
time = np.arange(10)

palette = gen_ok_wheel(n_colors=3)
for i in range(len(distance)):
    plt.plot(time, distance[i], color=palette[i])
plt.xlabel('Time m')
plt.ylabel('Distance m')
plt.legend(['walk', 'bike', 'car'])
plt.show()

# multi-parameter
x = np.linspace(-5, 5, 50)
beta = [0.5, 1, 2, 4]
y0 = 1/(1+np.exp(-x*beta[0]))
y1 = 1/(1+np.exp(-x*beta[1]))
y2 = 1/(1+np.exp(-x*beta[2]))
y3 = 1/(1+np.exp(-x*beta[3]))

palette = gen_ok_gradients(n_colors=4, init_hue=230)

plt.plot(x, y0, color=palette[0])
plt.plot(x, y1, color=palette[1])
plt.plot(x, y2, color=palette[2])
plt.plot(x, y3, color=palette[3])
plt.legend(['beta = 0.5', 'beta = 1', 'beta = 2', 'beta = 4'])
plt.show()  

#multi-parameter, multi-condition
def gen_brownian(y0, mu, sigma, t_steps):   
    dt = t_steps[1] - t_steps[0]
    drift = (mu - 0.5*sigma**2)*t_steps
    wiener_process = sigma*np.hstack([0, np.cumsum(np.sqrt(dt) * np.random.randn(len(t_steps)-1))])
    return y0+(np.exp(drift+wiener_process))-1

mu_s = [0.0, 0.5, 1]
n_trials = 5

palette = gen_ok_color_set(len(mu_s), n_trials, init_hue=30, shape=(len(mu_s), n_trials))

t = np.linspace(0, 1, 500)
for i in range(len(mu_s)):
    for j in range(n_trials):
        plt.plot(gen_brownian(0, mu_s[i], 0.15, t_steps=t), color=palette[i][j])
plt.show()


# transition from one color to another
red = gen_ok_color(30, 0.6)
blue = gen_ok_color(230, 0.6)

palette = ok_transition(red, blue, 500)

x = np.linspace(0, 1, len(palette))
for i in range(len(palette)):
    plt.vlines(x[i], 0, 1, color=palette[i])
plt.show()
"""
