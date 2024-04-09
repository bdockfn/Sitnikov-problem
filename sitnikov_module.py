import numpy as np
from scipy.integrate import odeint, solve_ivp


def time_moment(e, E):
    """
    Create time value from E, e.
    param e: eccentricity 
    param E: ecccentric anomaly \\
    return t = E-e*sin(e)
    """
    return E-e*np.sin(E)

def barycenter_distance_xy(a, e, E):
    """
    Calculate distance from barycenter to massive object in plane XY.
    param a: semi-major axis
    param e: eccentricity 
    param E: ecccentric_anomaly \\
    return r = a*(1-e*cos(E))
    """
    return a*(1-e*np.cos(E))

def derivative_z(r, v):
    """
    Derivative of z by eccentric anomaly.
    param r: barycenter distance
    param v: velocity \\
    return dz/dE
    """
    return 2*r*v

def derivative_v(r, z):
    """
    Derivative of z by eccentric anomaly.
    param r: barycenter distance in plane XY
    param z: barycenter distance along Z axis \\
    return dv/dE
"""
    return -2*r*z / np.sqrt(z**2 + r**2)**3


def ode_function(y, E, a, e):
    z, v = y
    r = barycenter_distance_xy(a, e, E)
    dz_dE = derivative_z(r, v)
    dv_dE = derivative_v(r, z)
    dy_dE = [dz_dE, dv_dE]
    return dy_dE


def sitnikov(y0, a, e, period, N, step):
    """
    param y0: z0, v0 \\ 
    param a: \\
    param e: \\
    param N: amount of periods \\
    param step: step between periods
    """
    E = np.arange(start=0, stop=period*N, step=step)
    res = odeint(ode_function, y0, E, args=(a, e))
    return(E, res)

def initial_grid_zv(start_z, stop_z, start_v, stop_v, step):
    z0 = np.arange(start_z, stop_z, step)
    v0 = np.arange(start_v, stop_v, step)

    grid_z0_v0 = [[[zi, vi] for vi in v0] for zi in z0]

    grid_z0_v0 = np.array(grid_z0_v0)
    grid_z0_v0 = np.reshape(grid_z0_v0, (len(z0)*len(v0), 2))

    return grid_z0_v0


def solution(y0, a, e, period, N, step):
    y = []
    for idx in range(y0.shape[0]):
        E, y_point = sitnikov(y0[idx], a, e, period, N, step)
        y.append(y_point)

    res = np.array(y)
    res = np.reshape(res, (res.shape[0]*res.shape[1], 2))
    return res


