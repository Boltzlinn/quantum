from lattice import Lattice
from quantum import sigma
import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jit
from functools import partial
import time

def hBN():
    sqrt_3 = np.sqrt(3.0)
    Pi = np.pi
    aH = 0.529
    Redberg = 13.6
    a0 = 1.45
    a = sqrt_3 * a0
    avec = np.array([[a * 0.5 * sqrt_3, a * 0.5], [a * 0.5 * sqrt_3, -a * 0.5]], np.float32)
    bvec = np.array([[(2.0 * Pi) / (sqrt_3 * a), (2.0 * Pi) / a], [(2.0 * Pi) / (sqrt_3 * a), -(2.0 * Pi) / a]], np.float32)
    hoppings = tuple([(G1, G2) for G1 in range(-3, 4) for G2 in range(-3, 4) if G1 ** 2 + G2 ** 2 - G1 * G2 <= 4])
    special_pts = {"$G$": (0., 0.), "$M$": (0.5, 0.5), "$K$": (1./3., 2./3.)}
    num_k1 = 10
    hBN = Lattice(2, avec, bvec, real_shape=num_k1, kstart=(0.5 / num_k1 - 0.5, 0.5 / num_k1 - 0.5), bz_shape=5, special_pts=special_pts, hoppings=hoppings)
    hBN.mk_basis({},{})
    Vs = np.array([10.785, 1.472, 7.8])
    Va = np.array([8.007, 0, 3.697])
    @jit
    def kinetic(k, d_qtm_nk, nd_qtm):
        kvec = (k + jnp.array(nd_qtm)) @ bvec
        return Redberg * aH ** 2 * (kvec @ kvec.T)
    hBN.kinetic = kinetic

    def ext_potential(d_qtm_nk, delta_nd_qtm):
        G1, G2 = delta_nd_qtm
        phase = (G1 + G2) * Pi / 3.
        G_norm = G1 ** 2 + G2 ** 2 - G1 * G2
        res = 0.
        if G_norm == 1:
            res = Vs[0] * np.cos(phase) + 1j * Va[0] * np.sin(phase)
        elif G_norm == 3:
            res = Vs[1] * np.cos(phase) + 1j * Va[1] * np.sin(phase)
        elif G_norm == 4:
            res = Vs[2] * np.cos(phase) + 1j * Va[2] * np.sin(phase)
        return res
        
    hBN.ext_potential = ext_potential
    hBN.mk_V()

    @jit
    def interaction(q, delta_d_qtm_nk):
        q_norm = jnp.linalg.norm(q @ bvec)
        res = jnp.where(q_norm == 0., 0., 2*Redberg*aH * 2 * Pi / q_norm / (1. + 10. * q_norm) / num_k1 ** 2 / np.abs(np.linalg.det(avec)) )
        return res
    
    hBN.interaction = interaction
    #hBN.plot_bands(["$G$", "$M$", "$K$"], num_pts=300, close=True)
    Omega = jnp.linspace(0,12,200)
    hBN.SHG(Omega, 1, 1, 1, 1)
    #hBN.SHG_ipa(Omega,[0,5])
    #print(hBN.hm(jnp.array((0.,0.)),()))
    #hBN.bse_solve(1,1,5)


if __name__ == "__main__":
    hBN()