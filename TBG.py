from lattice import Lattice
from quantum import sigma
import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jit
from functools import partial


def TBG():
    Pi = np.pi
    a0 = 2.46
    theta_angle = 1.05*Pi/180.0
    Ls = a0*0.5/np.sin(theta_angle*0.5)
    sqrt_3 = np.sqrt(3.0)
    a = Ls/sqrt_3
    Occ_factor = 4.0/8.0
    vF = 5.253
    u0 = 0.0797
    u0P = 0.0975
    mass = 0.015
    dim = 2
    avec = np.array(
        [[a*1.5, a*0.5*sqrt_3], [a*1.5, -a*0.5*sqrt_3]], np.float32)
    bvec = np.array([[(2.0 * Pi) / (3.0 * a), (2.0 * Pi) / (sqrt_3 * a)],
                      [(2.0 * Pi) / (3.0 * a), -(2.0 * Pi) / (sqrt_3 * a)]], np.float32)
    num_k1 = 1
    hoppings = ((0, 0), (1, 0), (-1, 0), (0, -1), (0, 1))
    special_pts = {"$G$": (0., 0.), "$M_1$": (0.5, 0.), "$M_2$": (
        0., 0.5), "$M_3$": (0.5, 0.5), "$K_1$": (1./3., 2./3.), "$K_2$": (2./3., 1./3.)}
    tbg = Lattice(dim, avec, bvec, real_shape=num_k1,
                  kstart=0.5 / num_k1, bz_shape=5, special_pts=special_pts, hoppings=hoppings, num_sub=4)
    tbg.mk_basis({"v": [-1, 1]}, {})
    # tbg.print_basis()
    @partial(jit, static_argnums=(1,2,3))
    def hopping_func(k, d_qtm_nk, nd_qtm1, delta_nd_qtm):
        vly, = d_qtm_nk
        G = np.array(nd_qtm1)
        U0 = np.array([[u0, u0P], [u0P, u0]], dtype=np.complex64)
        U1 = np.array([[u0, u0P * np.exp(1j * 2 * Pi / 3)],
                       [u0P * np.exp(-1j * 2 * Pi / 3), u0]], dtype=np.complex64)
        U2 = np.array([[u0, u0P * np.exp(-1j * 2 * Pi / 3)],
                       [u0P * np.exp(1j * 2 * Pi / 3), u0]], dtype=np.complex64)
        res = jnp.zeros((4, 4), dtype=jnp.complex64)
        if vly == 1:
            U1 = U1.conj()
            U2 = U2.conj()
        if delta_nd_qtm == (0, 0):
            k1 = (k + G - np.array([1. / 3., 2. / 3.])) @ bvec
            k2 = (k + G - np.array([2. / 3., 1. / 3.])) @ bvec
            if vly == 1:
                k1, k2 = k2, k1

            res = res.at[0:2, 0:2].set(-vF * k1[0] * vly * sigma[1] - vF * k1[1] * sigma[2] + mass * sigma[3])
            res = res.at[0:2, 2:4].set(U0.T.conj())
            res = res.at[2:4, 0:2].set(U0)
            res = res.at[2:4, 2:4].set(-vF * k2[0] * vly * sigma[1] - vF * k2[1] * sigma[2])
        elif delta_nd_qtm == tuple(vly * np.array((1, 0))):
            res = res.at[0:2, 2:4].set(U1.T.conj())
        elif delta_nd_qtm == tuple(-vly * np.array((1, 0))):
            res = res.at[2:4, 0:2].set(U1)
        elif delta_nd_qtm == tuple(-vly * np.array((0, 1))):
            res = res.at[0:2, 2:4].set(U2.T.conj())
        elif delta_nd_qtm == tuple(vly * np.array((0, 1))):
            res = res.at[2:4, 0:2].set(U2)
        return jnp.asarray(res)
    tbg.hopping_func = hopping_func

    # tbg.mk_hamiltonian()
    # tbg.print_hamiltonian()
    # tbg.solve()
    # tbg.print_eigen_energies()

    tbg.plot_bands(["$G$", "$M_3$", "$K_1$", "$G$", "$K_2$",
                    "$M_3$"], num_pts=300, close=False)


if __name__ == "__main__":
    TBG()
