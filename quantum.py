import jax.numpy as jnp
from jax import jacfwd, jit
import numpy as np
from functools import partial

sigma0 = np.eye(2, dtype=jnp.complex64)
sigmax = np.array([[0., 1.], [1., 0.]], dtype=np.complex64)
sigmay = np.array([[0., -1j], [1j, 0]], dtype=np.complex64)
sigmaz = np.array([[1., 0.], [0., -1.]], dtype=np.complex64)
sigmap = np.array([[0., 1.], [0., 0.]], dtype=np.complex64)
sigmam = np.array([[0., 0.], [1., 0.]], dtype=np.complex64)

sigma = np.array([sigma0, sigmax, sigmay, sigmaz, sigmap, sigmam])


def p_mat(i, j, N):
    res = np.zeros((N, N), dtype=np.complex64)
    res[i, j] = 1.
    return jnp.asarray(res)


class Quantum:

    def __init__(self, ifspin=0, ifmagnetic=0, hoppings=(), num_sub=1):
        self.diag_qtm_nums = {}
        self.ndiag_qtm_nums = {}
        self.hoppings = hoppings
        self.num_sub = num_sub
        if ifspin:
            if ifmagnetic:
                self.ndiag_qtm_nums['spin'] = [-0.5, 0.5]
            else:
                self.diag_qtm_nums['spin'] = [-0.5, 0.5]
    @staticmethod
    def fermi(e):
        if e > 0.:
            return 0
        else:
            return 1

    @staticmethod
    def hopping_func(d_qtm, nd_qtm1, delta_nd_qtm):
        pass

    def nd_qtm_add(self, nd_qtm1, delta_nd_qtm):
        nd_qtm2 = tuple(np.array(nd_qtm1) + np.array(delta_nd_qtm))
        j = self.nd_basis.get(nd_qtm2)
        return nd_qtm2, j
    
    def nd_qtm_minus(self, nd_qtm1, delta_nd_qtm):
        nd_qtm2 = tuple(np.array(nd_qtm1) - np.array(delta_nd_qtm))
        j = self.nd_basis.get(nd_qtm2)
        return nd_qtm2, j

    def mk_basis(self, diag_qtm_nums, ndiag_qtm_nums):
        self.diag_qtm_nums.update(diag_qtm_nums)
        if len(self.diag_qtm_nums) == 0:
            self.diag_qtm_nums = {'anon.': [0]}
        self.ndiag_qtm_nums.update(ndiag_qtm_nums)
        self.num_d_qtm = tuple(map(len, self.diag_qtm_nums.values()))
        self.num_nd_qtm = tuple(map(len, self.ndiag_qtm_nums.values()))
        self.num_nd_basis = int(np.product(self.num_nd_qtm))
        self.num_nd_bands = self.num_nd_basis * self.num_sub
        Basis = {}
        diag_Idx = [[]]
        diag_keys = self.diag_qtm_nums.keys()
        for key in diag_keys:
            diag_Idx = [idx+[idx1]
                        for idx in diag_Idx for idx1 in self.diag_qtm_nums[key]]
        ndiag_Idx = [[]]
        ndiag_keys = self.ndiag_qtm_nums.keys()
        for key in ndiag_keys:
            ndiag_Idx = [idx+[idx1]
                         for idx in ndiag_Idx for idx1 in self.ndiag_qtm_nums[key]]
        basis = {}
        j = 0
        for ndiag_idx in ndiag_Idx:
            basis[tuple(ndiag_idx)] = j
            j = j + 1
        self.nd_basis = basis
        i = 0
        for diag_idx in diag_Idx:
            Basis[tuple(diag_idx)] = (i, basis)
            i = i+1
        self.Basis = Basis

    def print_basis(self):
        diag_keys = list(self.diag_qtm_nums.keys())
        ndiag_keys = list(self.ndiag_qtm_nums.keys())
        temp = {}
        i = 0
        for diag_idx in self.Basis:
            for ndiag_idx in self.Basis[diag_idx][1]:
                temp[diag_idx+ndiag_idx] = i
                i = i+1
        print("The basis set {"+str(tuple(diag_keys+ndiag_keys))+":#} is:")
        print(temp)

    @partial(jit, static_argnums=(0, 1))
    def hm(self, d_qtm):
        basis = self.nd_basis
        num_sub = self.num_sub
        hopping_func = self.hopping_func
        res = jnp.zeros((self.num_nd_bands, self.num_nd_bands), dtype=jnp.complex64)
        for delta_nd_qtm in self.hoppings:
            # self.hopping could be tuple, list, etc. but not a generator
            i = 0
            for nd_qtm1 in basis:
                i = basis[nd_qtm1]
                j = self.nd_qtm_minus(nd_qtm1, delta_nd_qtm)[1]
                if j != None:
                    res[i * num_sub : (i + 1) * num_sub, j * num_sub : (j + 1) * num_sub] = hopping_func(d_qtm, nd_qtm1, delta_nd_qtm)
                i = i + 1
        return jnp.asarray(res)

    def mk_hamiltonian(self):
        hamiltonian = {}
        for d_qtm in self.Basis:
            hamiltonian[d_qtm] = self.hm(d_qtm)
        self.hamiltonian = hamiltonian

    def print_hamiltonian(self):
        print("The Hamiltonian of "+str(tuple(self.diag_qtm_nums.keys()))+" is:")
        for diag_idx in self.hamiltonian:
            print(diag_idx)
            print(jnp.round(self.hamiltonian[diag_idx], decimals=3)[0:4])

    def solve(self):
        eigen_energies = {}
        eigen_wvfuncs = {}
        for diag_idx in self.hamiltonian:
            eigs, eigvecs = jnp.linalg.eigh(
                self.hamiltonian[diag_idx])
            eigen_energies[diag_idx] = eigs
            eigen_wvfuncs[diag_idx] = eigvecs
        self.eigen_energies = eigen_energies
        self.eigen_wvfuncs = eigen_wvfuncs

    def print_eigen_energies(self):
        print("The eigen energies of " +
              str(tuple(self.diag_qtm_nums.keys()))+" is:")
        for diag_idx in self.eigen_energies:
            print(diag_idx)
            print(self.eigen_energies[diag_idx])


def test_Quantum():
    d_qtm = {'i': list(range(5))}
    nd_qtm = {'j': list(range(5))}
    hoppings = ((0.), (1.), (-1.))

    def hopping_func(d_qtm, nd_qtm1, delta_nd_qtm):
        if delta_nd_qtm == (0.):
            return 1.
        else:
            return 0.5
    cubic = Quantum(ifspin=1, ifmagnetic=0, hoppings=hoppings)
    cubic.hopping_func = hopping_func
    cubic.mk_basis(d_qtm, nd_qtm)
    cubic.print_basis()
    cubic.mk_hamiltonian()
    cubic.print_hamiltonian()
    cubic.solve()
    cubic.print_eigen_energies()


if __name__ == "__main__":
    test_Quantum()
