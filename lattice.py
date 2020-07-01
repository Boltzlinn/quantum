from quantum import Quantum, sigma, p_mat
import jax.numpy as jnp
from jax import grad, jacfwd, hessian, jit, vmap, custom_jvp
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from functools import partial
import time


class Lattice(Quantum):

    def __init__(self, dim, avec, bvec, real_shape=3, kstart=0., bz_shape=5, special_pts={}, ifspin=0, ifmagnetic=0, hoppings=(), num_sub=1):
        self.dim = dim
        self.avec = avec
        self.bvec = bvec
        self.special_pts = special_pts
        self.diag_qtm_nums = {}
        self.ndiag_qtm_nums = {}
        self.V = {}
        if np.sum(real_shape) <= 0:
            print('real_shape can not be 0 or negative, real_shape is set to 1 instead')
            real_shape = 1
        self.real_shape = np.zeros(dim, dtype=np.int32)+real_shape
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
    
    @staticmethod
    def kinetic(k, d_qtm_nk, nd_qtm):
        print("kinetic not set!")
        pass

    @staticmethod
    def ext_potential(d_qtm_nk, delta_nd_qtm):
        print("external potential not set")
        pass
    
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

    @staticmethod
    @jit
    def vec2arr(vec):
        return vec[:,None]-vec[None,:]

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
                j = self.nd_qtm_minus(nd_qtm1, delta_nd_qtm)[1]
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

    @partial(jit, static_argnums=(0,2,3,4))
    def shg_ipa(self, k, d_qtm_nk, Omega, bd_range):
        eta = 0.001
        bot, top = bd_range
        eigval, wvfunc = jnp.linalg.eigh(self.hm(k, d_qtm_nk))
        Vel = jacfwd(self.hm, argnums=(0))(k, d_qtm_nk)
        vel = jnp.einsum('lk,lma,mn->kna', wvfunc.conj()[:, bot:top], Vel, wvfunc[:, bot:top])  #[bd_range,bd_range,dim]

        dist = jnp.where(eigval[bot:top] > 0., 0, 1)
        d_dist = self.vec2arr(dist)  #[bd_range,bd_range]
        d_en = self.vec2arr(eigval[bot:top])  #[bd_range,bd_range]

        r_inter = -1j * vel/d_en[:,:,None]  #[bd_range,bd_range,dim]
        r_inter = jnp.where(jnp.isinf(r_inter), 0.,r_inter)  #[bd_range,bd_range,dim]

        d_en_jac = vmap(self.vec2arr, 1, 2)(vmap(jnp.diag,2,1)(vel))  #[bd_range,bd_range,dim]
        
        r_inter_gd = (r_inter[:,:,:, None] * jnp.transpose(d_en_jac, axes=(1, 0, 2))[:,:, None,:] + r_inter[:,:, None,:] * (jnp.transpose(d_en_jac, axes=(1, 0, 2))[:,:,:, None]) + 1j * jnp.einsum('nla,lmb->nmab', r_inter, r_inter * d_en[:,:, None]) - 1j * jnp.einsum('nlb,lma->nmab', r_inter * d_en[:,:, None], r_inter)) / d_en[:,:, None, None]
        r_inter_gd = jnp.where(jnp.isinf(r_inter_gd), 0.,r_inter_gd)  #[bd_range,bd_range,dim,dim]

        f = r_inter[None,:,:,:] / (Omega[:,None,None,None] + 1j * eta - d_en[None,:,:,None])  #[nOmega,bd_range,bd_range,dim]
        h = r_inter[None,:,:,:] / (2 * Omega[:, None, None, None] + 2 * 1j * eta + d_en[None,:,:, None])  #[nOmega,bd_range,bd_range,dim]
        
        g = d_dist.T[None,:,:, None] * f  #[nOmega,bd_range,bd_range,dim]
        g_gd = (d_dist.T[None,:,:, None, None] * r_inter_gd[None,:,:,:,:] + g[:,:,:, None,:] * d_en_jac[None,:,:,:, None]) / (Omega[:, None, None, None, None] + 1j * eta - d_en[None,:,:, None, None])  #[nOmega,bd_range,bd_range,dim,dim]
        
        rg = 1j * g_gd + jnp.einsum('nlb,olma->onmba', r_inter, g) - jnp.einsum('onla,lmb->onmba', g, r_inter)  #[nOmega,bd_range,bd_range,dim,dim]
        res = jnp.transpose(h, (0, 2, 1, 3))[:,:,:,:, None, None] * rg[:,:,:, None,:,:] - 1j / 2 * jnp.transpose(f, (0, 2, 1, 3))[:,:,:, None, None,:] * g_gd[:,:,:,:,:, None]  #[nOmega,bd_range,bd_range,dim,dim,dim]
        return jnp.sum(res, axis=(1, 2))

    def mk_hamiltonian(self):
        hamiltonian = {}
        for d_qtm in self.Basis:
            k, d_qtm_nk = self.separate(d_qtm, self.dim)
            hamiltonian[d_qtm] = self.hm(k, d_qtm_nk)
        self.hamiltonian = hamiltonian

    def SHG_ipa(self, Omega, bd_range):
        dim = self.dim
        num_O = len(Omega)
        res = jnp.zeros((num_O, dim, dim, dim), dtype=jnp.complex64)
        flag = 0
        for d_qtm in self.Basis:
            k, d_qtm_nk = self.separate(d_qtm, dim)
            k = jnp.array(k)
            res = res + self.shg_ipa(k, d_qtm_nk, Omega, bd_range)
            print(flag)
            flag = flag + 1

        plt.plot(Omega, jnp.abs(res[:,0,0,0])/flag*Omega)
        plt.savefig("temp_shg.pdf")
        plt.show()

    def plot_bands(self, pts_list, num_pts=200, d_qtm_nk=None, close=True):
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
        plt.savefig("temp_band.pdf")
        plt.show()


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
