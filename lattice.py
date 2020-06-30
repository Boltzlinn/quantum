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
        self.flag = 0
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

    def separate(self, d_qtm):
        k = np.array(d_qtm[0:self.dim])
        if len(d_qtm) == self.dim:
            d_qtm_nk = ()
        else:
            d_qtm_nk = d_qtm[self.dim:]
        return k, d_qtm_nk

    @partial(jit, static_argnums=(0))
    def mk_V(self):
        d_qtm_list = list(self.Basis.keys())
        d_qtm_nk_list = list(set(map(lambda d_qtm: self.separate(d_qtm)[1],d_qtm_list)))

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
    
    @partial(jit, static_argnums=(0))
    def hm(self, k, d_qtm_nk):
        num_nd_bands = self.num_nd_bands
        num_sub = self.num_sub
        basis = self.nd_basis
        num_nd_basis = self.num_nd_basis
        kinetic = self.kinetic
        res = jnp.zeros((num_nd_bands, num_nd_bands), dtype=jnp.complex64)
        #diag = jnp.kron(jnp.eye(num_nd_basis, dtype=jnp.complex64), jnp.ones((num_sub, num_sub), dtype=jnp.complex64))
        #data = []
        for nd_qtm in basis:
            i = basis[nd_qtm]
            res = res + jnp.kron(p_mat(i, i, num_nd_basis), kinetic(k, d_qtm_nk, nd_qtm))
            #data = data + [kinetic(k, d_qtm_nk, nd_qtm)]
        #T = jnp.repeat(jnp.vstack(tuple(data)),num_nd_basis,axis=1) * diag
        res = res + self.V[d_qtm_nk]
        return jnp.asarray(res)

    @partial(jit, static_argnums=(0,3,4))
    def shg_ipa(self, k, d_qtm_nk, Omega, bd_range):
        eta = 0.03
        bot, top = bd_range
        num_bds = top - bot
        Res = lambda kk: jnp.linalg.eigh(self.hm(kk, d_qtm_nk))
        Jac = jacfwd(Res)
        #Hess = jacfwd(Jac)
        eigval, wvfunc = Res(k)
        eigval_jac, wvfunc_jac = Jac(k)
        #eigval_hess, wvfunc_hess = Hess(k)
        #print(3)
        dist = jnp.where(eigval[bot:top] > 0., 0, 1)
        def vec2arr(vec):
            return vec[:,None]-vec[None,:]
        f = vec2arr(dist)  #[bd_range,bd_range]
        e = vec2arr(eigval[bot:top])  #[bd_range,bd_range]
        no_diag = jnp.ones((num_bds, num_bds)) - jnp.eye(num_bds)
        xi = vmap(jnp.dot, (None, 2), 2)(wvfunc.T.conj()[bot:top], wvfunc_jac[:, bot:top])  #[bd_range,bd_range,dim]
        re = no_diag[:,:,None] * xi  #[bd_range,bd_range,dim]
        ri = vmap(jnp.diag, 2, 1)(xi)  #[bd_range,dim]
        ri = vmap(vec2arr, 1, 2)(ri)  #[bd_range,bd_range,dim]
        e_jac = vmap(vec2arr, 1, 2)(eigval_jac[bot:top])  #[bd_range,bd_range,dim]
        e_gd = e_jac - 1j * e[:,:,None] * ri  #[bd_range,bd_range,dim]
        #xi_jac = vmap(vmap(jnp.dot, (2, None), 2), (None, 2), 3)(jnp.transpose(wvfunc_jac[:, bot:top], axes=(1, 0, 2)).conj(), wvfunc_jac[:, bot:top]) + vmap(vmap(jnp.dot, (None, 2), 2), (None, 3), 3)(wvfunc.T.conj()[bot:top], wvfunc_hess[:, bot:top])  #[bd_range,bd_range,dim,dim]
        #re_jac = no_diag[:,:, None, None] * xi_jac  #[bd_range,bd_range,dim,dim]
        re_gd = (re[:,:,:, None] * (jnp.transpose(e_gd, axes=(1, 0, 2))[:,:, None,:]) + re[:,:, None,:] * (jnp.transpose(e_gd, axes=(1, 0, 2))[:,:,:, None]) + 1j * jnp.einsum('nla,lmb->nmab', re, re * e[:,:, None]) - 1j * jnp.einsum('nlb,lma->nmab', re * e[:,:, None], re)) / e[:,:, None, None]
        re_gd = jnp.where(jnp.isinf(re_gd), 0.,re_gd)
        #re_gd = re_jac - 1j * re[:,:, None,:] * ri[:,:,:, None]  #[bd_range,bd_range,dim,dim]
        g = f.T[None,:,:, None] * re[None,:,:,:] / (Omega[:, None, None, None] + 1j * eta - e[None,:,:, None])  #[nOmega,bd_range,bd_range,dim]
        gg = re[None,:,:,:] / (Omega[:,None,None,None] + 1j * eta - e[None,:,:,None])  #[nOmega,bd_range,bd_range,dim]
        h = re[None,:,:,:] / (2 * Omega[:,None,None,None] + 2 * 1j * eta + e[None,:,:,None])  #[nOmega,bd_range,bd_range,dim]
        g_gd = f.T[None,:,:,None,None] * (re_gd[None,:,:,:,:] + re[None,:,:,None,:] * e_gd[None,:,:,:,None] / (Omega[:,None,None,None,None] + 1j * eta - e[None,:,:,None,None])) / (Omega[:,None,None,None,None] + 1j * eta - e[None,:,:,None,None])  #[nOmega,bd_range,bd_range,dim,dim]
        rg = 1j * g_gd + jnp.einsum('nlb,olma->onmba', re, g) - jnp.einsum('onla,lmb->onmba', g, re)  #[nOmega,bd_range,bd_range,dim,dim]
        res = jnp.transpose(h, (0, 2, 1, 3))[:,:,:,:, None, None] * rg[:,:,:, None,:,:] - 1j / 2 * jnp.transpose(gg, (0, 2, 1, 3))[:,:,:, None, None,:] * g_gd[:,:,:,:,:, None]
        return jnp.sum(res, axis=(1, 2))

    def mk_hamiltonian(self):
        hamiltonian = {}
        for d_qtm in self.Basis:
            k, d_qtm_nk = self.separate(d_qtm)
            hamiltonian[d_qtm] = self.hm(k, d_qtm_nk)
        self.hamiltonian = hamiltonian

    def SHG_ipa(self, Omega, bd_range):
        dim = self.dim
        num_O = len(Omega)
        res = jnp.zeros((num_O, dim, dim, dim), dtype=jnp.complex64)
        flag=0
        for d_qtm in self.Basis:
            k, d_qtm_nk = self.separate(d_qtm)
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
            d_qtm_nk_list = list(set(map(lambda d_qtm: self.separate(d_qtm)[1],d_qtm_list)))
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
