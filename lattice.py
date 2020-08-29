from quantum import Quantum, sigma, commutator
import jax.numpy as jnp
from jax import grad, jacfwd, hessian, jit, vmap, custom_jvp
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from functools import partial
from timeit import timeit
from operator import itemgetter
import time
import sys


class Lattice(Quantum):

    def __init__(self, dim, avec, bvec, real_shape=3, kstart=0., bz_shape=5, special_pts={}, ifspin=0, ifmagnetic=0, hoppings=(), num_sub=1, name = None):
        self.dim = dim
        self.avec = avec
        self.vcell = np.abs(np.linalg.det(avec))
        self.bvec = bvec
        self.special_pts = special_pts
        self.name = name
        self.diag_qtm_nums = {}
        self.ndiag_qtm_nums = {}
        self.V = {}
        if np.sum(real_shape) <= 0:
            print('real_shape can not be 0 or negative, real_shape is set to 1 instead')
            real_shape = 1
        self.real_shape = np.zeros(dim, dtype=np.int32) + real_shape
        self.volume = self.vcell * np.prod(self.real_shape)
        self.kstart = np.zeros(dim)+kstart
        if np.sum(bz_shape) <= 0:
            print('bz_shape can not be 0 or negative, bz_shape is set to 5 instead')
            bz_shape = 5
        self.bz_shape = np.zeros(dim, dtype=np.int32) + bz_shape
        self.hoppings = hoppings
        self.num_sub = num_sub
        klabels = ('kx', 'ky', 'kz')
        Glabels = ('Gz', 'Gy', 'Gz')
        for i in range(dim):
            self.diag_qtm_nums[klabels[i]] = [
                self.kstart[i] + n / self.real_shape[i] for n in range(self.real_shape[i])]
            self.ndiag_qtm_nums[Glabels[i]] = [
                n - self.bz_shape[i] // 2 for n in range(self.bz_shape[i])]
        if ifspin:
            if ifmagnetic:
                self.ndiag_qtm_nums['spin'] = [-0.5, 0.5]
            else:
                self.diag_qtm_nums['spin'] = [-0.5, 0.5]
    
    def basis_to_arr(self):
        self.d_qtm_list = list(self.Basis.keys())
        self.d_qtm_array = np.array(self.d_qtm_list, dtype=np.float32)
        self.G_list = list(set(map(lambda nd_qtm: self.separate(nd_qtm, self.dim)[0], self.nd_basis)))
        self.G_array = np.array(self.G_list, dtype=np.float32)
        self.d_qtm_nk_list = list(set(map(lambda d_qtm: self.separate(d_qtm, self.dim)[1], self.d_qtm_list)))

    def set_kinetic(self, kinetic):
        self.kinetic = kinetic

    def set_ext_potential(self, ext_potential):
        self.ext_potential = ext_potential

    def set_interaction(self, interaction):
        self.interaction = interaction
        self.interaction_grad = jacfwd(self.interaction, argnums=(0))
    
    @staticmethod
    def separate(container, idx):
        l = len(container)
        if idx > l:
            part1 = container
            part2 = ()
            print("index larger than container length, the second part is set ()")
        elif idx == l:
            part1 = container
            part2 = ()
        else:
            part1 = container[0:idx]
            part2 = container[idx:]
        return part1, part2

    @partial(jit, static_argnums=(0))
    def mk_V(self):
        d_qtm_list = list(self.Basis.keys())
        d_qtm_nk_list = list(set(map(lambda d_qtm: self.separate(d_qtm, self.dim)[1],d_qtm_list)))

        basis = self.nd_basis
        num_sub = self.num_sub
        V = {d_qtm_nk: np.zeros((self.num_nd_bands, self.num_nd_bands), dtype=np.complex64) for d_qtm_nk in d_qtm_nk_list}
        for delta_nd_qtm in self.hoppings:
            # self.hopping could be tuple, list, etc. but not a generator
            for nd_qtm1 in basis:
                i = basis.get(nd_qtm1)
                nd_qtm2 = self.qtm_minus(nd_qtm1, delta_nd_qtm)
                j = self.nd_basis.get(nd_qtm2)
                if j != None:
                    for d_qtm_nk in V:
                        V[d_qtm_nk][i * num_sub:(i + 1) * num_sub, j * num_sub:(j + 1) * num_sub] = self.ext_potential(d_qtm_nk, delta_nd_qtm)
        self.V = V
    
    @partial(jit, static_argnums=(0, 2))
    def hm(self, k, d_qtm_nk):
        data = tuple(map(lambda nd_qtm: self.kinetic(k, d_qtm_nk, nd_qtm), list(self.nd_basis.keys())))
        diag = jnp.kron(jnp.eye(self.num_nd_basis, dtype=jnp.complex64), jnp.ones((self.num_sub, self.num_sub), dtype=jnp.complex64))
        T = jnp.repeat(jnp.hstack(data).reshape((1, -1)), self.num_nd_basis, axis=0).reshape((self.num_nd_bands, self.num_nd_bands)) * diag
        return jnp.asarray(T + self.V[d_qtm_nk])

    def mk_hamiltonian(self):
        hamiltonian = []
        for d_qtm in self.Basis:
            k, d_qtm_nk = self.separate(d_qtm, self.dim)
            hamiltonian += [self.hm(np.asarray(k,dtype=np.float32), d_qtm_nk)]
        self.hamiltonian = np.asarray(hamiltonian, dtype=np.complex64)
        print('non-interacting hamiltonian made')

    #@partial(jit, static_argnums=(0,1,2,3))
    def mk_r_bare(self, nv, nf, nc):
        vel_opt = jit(jacfwd(self.hm, argnums=(0)), static_argnums=(1))
        vel_plane_wave = []
        for d_qtm in self.Basis:
            k, d_qtm_nk = self.separate(d_qtm, self.dim)
            vel_plane_wave += [vel_opt(np.asarray(k,dtype=np.float32), d_qtm_nk)]
        vel_plane_wave = np.transpose(np.asarray(vel_plane_wave, dtype=np.complex64), axes=(3, 0, 1, 2))
        vel = np.transpose(self.wv_funcs.conj(), axes=(0,2,1)) @ vel_plane_wave @ self.wv_funcs #asij
        epsilon = self.energies[:,:, None] - self.energies[:, None]
        epsilon_inv = np.asarray(1./(epsilon + np.diag(np.inf * np.ones((self.num_nd_bands)))), dtype=np.float32)
        r_bare = -1j * vel * epsilon_inv
        energies_grad=np.einsum('asii->asi', vel)
        epsilon_gd=energies_grad[:,:,:, None] - energies_grad[:,:, None,:]  #asij
        r_bare_gd=(commutator(r_bare[:, None], vel) - epsilon_gd[:, None] * r_bare) * epsilon_inv
        r_bare = np.transpose(r_bare[:,:, nf - nv:nf + nc, nf - nv:nf + nc], axes=(1, 2, 3, 0))
        r_bare_cv = np.asarray(r_bare[:, nv:nv + nc, 0:nv].reshape((-1, self.dim)), order='C')
        vel_bare = np.transpose(vel[:,:, nf - nv:nf + nc, nf - nv:nf + nc], axes=(1, 2, 3, 0))
        vel_bare_cv = np.asarray(vel_bare[:, nv:nv + nc, 0:nv].reshape((-1, self.dim)), order='C')
        self.r_bare = np.asarray(np.transpose(r_bare, axes=(3, 0, 1, 2)), order='F')
        self.vel_bare = np.asarray(np.transpose(vel_bare, axes=(3, 0, 1, 2)), order='F')
        self.r_bare_gd=np.asarray(r_bare_gd[:,:,:, nf - nv:nf + nc, nf - nv:nf + nc],order='F')
        self.epsilon=np.asarray(epsilon[:, nf - nv:nf + nc, nf - nv:nf + nc],order='F')
        self.epsilon_gd=np.asarray(epsilon_gd[:,:, nf - nv:nf + nc, nf - nv:nf + nc],order='F')
        self.r_bare_cv = r_bare_cv
        self.vel_bare_cv = vel_bare_cv
        print("r_bare solved")

    #@partial(jit, static_argnums=(0))
    def mk_overlape_idx(self):
        num_sub = self.num_sub
        def task(G):
            idx1 = []
            idx2 = []
            for nd_qtm1 in self.nd_basis:
                i = self.nd_basis[nd_qtm1]
                l = len(nd_qtm1) - self.dim
                delta_nd_qtm = G + tuple((0 for i in range(l)))
                nd_qtm2 = self.qtm_minus(nd_qtm1, delta_nd_qtm)
                j = self.nd_basis.get(nd_qtm2)
                if j != None:
                    idx1 += list(range(i * num_sub, (i + 1) * num_sub))
                    idx2 += list(range(j * num_sub, (j + 1) * num_sub))
            return [idx1, idx2]
        self.overlape_idx = np.array([task(G) for G in self.G_list])

    def plot_bands(self, pts_list, num_pts=200, d_qtm_nk=None, close=True, energy_lim = None):
        if len(pts_list) == 0:
            print("No points to plot")
            pass
        if len(self.Basis) == 0:
            print("Please set basis first")
            pass
        else:
            if len(self.nd_basis) == 0:
                print("Please set basis first")
                pass

        d_qtm_nk_list = []
        if d_qtm_nk == None:
            d_qtm_list = list(self.Basis.keys())
            d_qtm_nk_list = list(set(map(lambda d_qtm: self.separate(d_qtm, self.dim)[1],d_qtm_list)))
        else:
            d_qtm_nk_list = [d_qtm_nk]

        label_list = []
        vec_list = []
        if isinstance(pts_list, dict):
            self.special_pts.update(pts_list)
            label_list = list(pts_list.keys())
            vec_list = list(pts_list.values())
        elif isinstance(pts_list, (list, tuple)):
            for label in pts_list:
                if self.special_pts.get(label) == None:
                    print("Undefined point "+label)
                    pass

            label_list = list(pts_list)
            vec_list = [self.special_pts[label] for label in label_list]
        if close:
            label_list = label_list + [label_list[0]]
            vec_list = vec_list + [vec_list[0]]

        vec_list = np.array(vec_list)
        num_nodes = len(vec_list)
        len_list = np.array([np.linalg.norm(
            (vec_list[i+1]-vec_list[i])@self.bvec) for i in range(num_nodes-1)])
        num_list = (len_list/len_list.sum()*num_pts).astype(np.int32)
        num_list[-1] = num_list[-1]+(num_pts-num_list.sum())
        node_list = np.insert(num_list.cumsum(), 0, 0)*len_list.sum()/num_pts
        kpath = []
        for i in range(num_nodes-1):
            kpath = kpath+[x/num_list[i]*vec_list[i+1] +
                           (1-x/num_list[i])*vec_list[i] for x in range(num_list[i])]

        x = np.linspace(0, len_list.sum(), num_pts)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        linestyles = ['-', '--', '-.', ':']
        for temp in zip(d_qtm_nk_list, linestyles):
            d_qtm_nk, lt = temp
            t0 = time.clock()
            hms = [self.hm(k, d_qtm_nk) for k in kpath]
            t1 = time.clock()
            engies = jnp.asarray([jnp.linalg.eigvalsh(mat) for mat in hms])
            t2 = time.clock()
            print(t1 - t0)
            print(t2 - t1)
            #engies = jnp.asarray([jnp.linalg.eigvalsh(self.hm(k, d_qtm_nk)) for k in kpath])
            [ax.plot(x, engies[:,i], lt) for i in range(self.num_nd_bands)]
        en_bot = np.min(engies[:,0])
        en_top = np.max(engies[:,-1])
        [ax.plot([node_list[i], node_list[i]], [en_bot, en_top], 'k--')
         for i in range(1, num_nodes-1)]
        ax.set_xticks(node_list)
        ax.set_xticklabels(label_list, minor=False)
        if energy_lim is not None: ax.set_ylim(energy_lim)
        plt.savefig(self.name+"_band.pdf")
        #plt.show()


def test_Lattice():
    dim = 2
    avec = sigma[0]
    bvec = sigma[0]
    cubic = Lattice(dim, avec, bvec,
                    real_shape=1, kstart=0, bz_shape=5)
    cubic.mk_basis({}, {})
    cubic.print_basis()
    hoppings = tuple((i, j) for i in range(-1, 2) for j in range(-1, 2))
    cubic.hoppings = hoppings

    def hopping_func(k, d_qtm_nk, nd_qtm1, delta_nd_qtm):
        if delta_nd_qtm == (0., 0.):
            return jnp.linalg.norm((k+jnp.array(nd_qtm1))@bvec)**2+4./(jnp.linalg.norm(jnp.array(delta_nd_qtm))+5)
        else:
            return 4./(jnp.linalg.norm(jnp.array(delta_nd_qtm))+5)
    cubic.hopping_func = hopping_func
    cubic.mk_hamiltonian()
    cubic.print_hamiltonian()
    # cubic.solve()
    # cubic.print_eigen_energies()
    cubic.plot_bands({"G": (0., 0.), "M": (0.5, 0.), "K": (0.5, 0.5)})

if __name__ == "__main__":
    test_Lattice()
