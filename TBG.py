from lattice import Lattice
from quantum import sigma
import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jit
from functools import partial
import time

def TBG():
    Pi = np.pi
    a0 = 2.46
    theta_angle = 1.05*Pi/180.0
    Ls = a0*0.5/np.sin(theta_angle*0.5)
    sqrt_3 = np.sqrt(3.0)
    a = Ls/sqrt_3
    vF = 5.253
    u0 = 0.0797
    u0P = 0.0975
    mass = 0.015
    dim = 2
    avec = np.array([[a * 1.5, a * 0.5 * sqrt_3], [a * 1.5, -a * 0.5 * sqrt_3]], np.float32)
    bvec = np.array([[(2.0 * Pi) / (3.0 * a), (2.0 * Pi) / (sqrt_3 * a)], [(2.0 * Pi) / (3.0 * a), -(2.0 * Pi) / (sqrt_3 * a)]], np.float32)
    num_k1 = 32
    hoppings = ((0, 0), (1, 0), (-1, 0), (0, -1), (0, 1))
    special_pts = {"$G$": (0., 0.), "$M_1$": (0.5, 0.), "$M_2$": (
        0., 0.5), "$M_3$": (0.5, 0.5), "$K_1$": (1./3., 2./3.), "$K_2$": (2./3., 1./3.)}
    tbg = Lattice(dim, avec, bvec, real_shape=num_k1,
                  kstart=0.5 / num_k1, bz_shape=5, special_pts=special_pts, hoppings=hoppings, num_sub=4)
    tbg.mk_basis({"v": [-1, 1]}, {})
    # tbg.print_basis()
    U0 = np.kron(sigma[5], np.array([[u0, u0P], [u0P, u0]], dtype=np.complex64))
    U1 = np.kron(sigma[5], np.array([[u0, u0P * np.exp(1j * 2 * Pi / 3)], [u0P * np.exp(-1j * 2 * Pi / 3), u0]], dtype=np.complex64))
    U2 = np.kron(sigma[5], np.array([[u0, u0P * np.exp(-1j * 2 * Pi / 3)], [u0P * np.exp(1j * 2 * Pi / 3), u0]], dtype=np.complex64))
    @jit
    def kinetic(k, d_qtm_nk, nd_qtm):
        vly, = d_qtm_nk
        G = jnp.array(nd_qtm)
        K1, K2 = np.array([[1. / 3., 2. / 3.], [2. / 3., 1. / 3.]])
        K1, K2 = jnp.where(vly == 1, (K2, K1), (K1, K2))
        k1 = (k + G - K1) @ bvec
        k2 = (k + G - K2) @ bvec
        res = jnp.kron((sigma[0] + sigma[3]) / 2, -vF * k1[0] * vly * sigma[1] - vF * k1[1] * sigma[2]) + jnp.kron((sigma[0] - sigma[3]) / 2, -vF * k2[0] * vly * sigma[1] - vF * k2[1] * sigma[2])
        return res
    tbg.kinetic = kinetic
    
    def ext_potential(d_qtm_nk, delta_nd_qtm):
        vly, = d_qtm_nk
        res = np.zeros((4, 4), dtype=np.complex64)
        if delta_nd_qtm == (0, 0):
            res = U0 + U0.T.conj() + np.kron((sigma[0] + sigma[3]) / 2,  mass * sigma[3])
        elif delta_nd_qtm == (1, 0):
            res = np.where(vly == 1, U1.T, U1)
        elif delta_nd_qtm == (-1, 0):
            res = np.where(vly == 1, U1.conj(), U1.T.conj())
        elif delta_nd_qtm == (0, 1):
            res = np.where(vly == 1, U2.conj(), U2.T.conj())
        elif delta_nd_qtm == (0, -1):
            res = np.where(vly == 1, U2.T, U2)
        return res
    tbg.ext_potential = ext_potential
    tbg.mk_V()

    epsilon = 2
    Kappa = 0.005
    Uvalue = 0.1071/epsilon  ## Uvalue=e^2/(4*pi*epsilon0*Ls) 
    Jvalue = 0.0017/epsilon ## Jvalue = Uvalue*(qM/|K-K'|)
    qM = 4*Pi/(np.sqrt(3.0)*Ls)
    @jit
    def interaction(q, d_qtm_nk1, d_qtm_nk2):
        q_norm = jnp.linalg.norm(q @ bvec)
        flag = jnp.all(d_qtm_nk1==d_qtm_nk2)
        res = jnp.where(q_norm == 0., 0., jnp.where(flag, Uvalue * qM / jnp.sqrt(q_norm ** 2 + Kappa ** 2), Jvalue))
        return res/num_k1**2
    tbg.interaction = interaction
    Omega = np.linspace(0,0.2,200)
    tbg.SHG(Omega, 2, 50, 2, eta=0.002)

    # tbg.mk_hamiltonian()
    # tbg.print_hamiltonian()
    # tbg.solve()
    # tbg.print_eigen_energies()

    #tbg.plot_bands(["$G$", "$M_3$", "$K_1$", "$G$", "$K_2$","$M_3$"], num_pts=300, close=False)


if __name__ == "__main__":
    TBG()
