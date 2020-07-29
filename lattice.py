from quantum import Quantum, sigma, p_mat
import jax.numpy as jnp
from jax import grad, jacfwd, hessian, jit, vmap, custom_jvp
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from functools import partial
import time
from operator import itemgetter


class Lattice(Quantum):

    def __init__(self, dim, avec, bvec, real_shape=3, kstart=0., bz_shape=5, special_pts={}, ifspin=0, ifmagnetic=0, hoppings=(), num_sub=1):
        self.dim = dim
        self.avec = avec
        self.bvec = bvec
        self.special_pts = special_pts
        self.diag_qtm_nums = {}
        self.ndiag_qtm_nums = {}
        self.hamiltonian = {}
        self.G_list = {}
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
    def interaction(q, delta_d_qtm_nk):
        print("interaction not set")
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
        hamiltonian = {}
        for d_qtm in self.Basis:
            k, d_qtm_nk = self.separate(d_qtm, self.dim)
            k = jnp.array(k)
            hamiltonian[d_qtm] = self.hm(k, d_qtm_nk)
        self.hamiltonian = hamiltonian

    @partial(jit, static_argnums=(0))
    def mk_overlape_idx(self):
        idx = []
        num_sub = self.num_sub
        eye = np.eye(self.num_nd_bands, dtype=np.int32)
        for G in self.G_list:
            idx1 = []
            idx2 = []
            for nd_qtm1 in self.nd_basis:
                i = self.nd_basis[nd_qtm1]
                l = len(nd_qtm1) - self.dim
                delta_nd_qtm = G + tuple((0 for i in range(l)))
                nd_qtm2 = self.qtm_minus(nd_qtm1, delta_nd_qtm)
                j = self.nd_basis.get(nd_qtm2)
                if j != None:
                    idx1 = idx1 + list(range(i * num_sub, (i + 1) * num_sub))
                    idx2 = idx2 + list(range(j * num_sub, (j + 1) * num_sub))
            idx = idx + [eye[idx1].T @ eye[idx2]]
        self.overlape_idx = np.asarray(idx)

    @partial(jit, static_argnums=(0))
    def hartree(self, d_qtm_nk1, d_qtm_nk2, wv_func1, wv_func2):
        interaction = vmap(partial(self.interaction, d_qtm_nk1 = d_qtm_nk1, d_qtm_nk2 = d_qtm_nk2))(self.G_array)
        overlape_1 = jnp.einsum('ei,gef,fj->gij', wv_func1.conj(), self.overlape_idx, wv_func1)
        overlape_2 = jnp.einsum('ei,gef,fj->gij', wv_func2.conj(), self.overlape_idx, wv_func2)
        res = jnp.einsum('g,gij,gkl->ijkl', interaction, overlape_1, overlape_2.conj())
        return res
    
    @partial(jit, static_argnums=(0))
    def fock(self, delta_k, d_qtm_nk1, d_qtm_nk2, wv_func1, wv_func2):
        q_array = delta_k + self.G_array
        interaction = vmap(partial(self.interaction, d_qtm_nk1 = d_qtm_nk1, d_qtm_nk2 = d_qtm_nk2))(q_array)
        interaction_grad = vmap(partial(self.interaction_grad, d_qtm_nk1 = d_qtm_nk1, d_qtm_nk2 = d_qtm_nk2))(q_array)
        interaction_val_grad = jnp.append(interaction.reshape((-1,1)), interaction_grad, axis=1)
        overlape = jnp.einsum('ei,gef,fj->gij', wv_func1.conj(), self.overlape_idx, wv_func2)
        res_val_grad = jnp.einsum('ga,gik,gjl->ijkla', interaction_val_grad, overlape, overlape.conj())
        return res_val_grad

    @partial(jit, static_argnums=(0))
    def W(self, delta_k, d_qtm_nk1, d_qtm_nk2, wv_func1, wv_func2):
        hartree = self.hartree(d_qtm_nk1, d_qtm_nk2, wv_func1, wv_func2)
        fock_val_grad = self.fock(delta_k, d_qtm_nk1, d_qtm_nk2, wv_func1, wv_func2)
        res = fock_val_grad - hartree[:,:,:,:, None] * jnp.append(0, jnp.ones(self.dim))[None, None, None, None,:]
        return res
    
    @partial(jit, static_argnums=(0))
    def mk_W(self):
        nv, nc, nf = self.nv, self.nc, self.nf
        d_qtm_nks = self.d_qtm_array[:,self.dim:]
        delta_d_qtms = self.d_qtm_array[:, None,:] - self.d_qtm_array[None,:,:]
        delta_ks = delta_d_qtms[:,:, 0:self.dim]
        wv_funcs = self.wv_funcs[:,:, nf - nv:nf + nc]
        def task(delta_ki,d_qtm_nki,wv_funci):
            return vmap(self.W, (0, None, 0, None, 0))(delta_ki, d_qtm_nki, d_qtm_nks, wv_funci, wv_funcs)
        W_val_grad = np.array(vmap(task, (0, 0, 0))(delta_ks, d_qtm_nks, self.wv_funcs))
        self.W_val_grad = W_val_grad
        print(np.shape(W_val_grad))
        print("W calculated")
    
    @partial(jit, static_argnums=(0))
    def mk_r_bare(self):
        nv, nc, nf = self.nv, self.nc, self.nf
        vel_plane_wave = []
        for d_qtm in self.Basis:
            k, d_qtm_nk = self.separate(d_qtm, self.dim)
            vel_plane_wave = vel_plane_wave + [self.vel_opt(jnp.array(k), d_qtm_nk)]
        vel_plane_wave = jnp.array(vel_plane_wave)
        vel = jnp.einsum('slk,slma,smn->skna', self.wv_funcs.conj(), vel_plane_wave, self.wv_funcs)
        epsilon = vmap(self.vec2arr)(self.energies)
        r_bare = -1j * vel / epsilon[:,:,:, None]
        r_bare = jnp.where(jnp.isinf(r_bare), 0.,r_bare)
        energies_grad=jnp.einsum('siia->sia', vel)
        epsilon_gd = energies_grad[:,:,None,:]-energies_grad[:,None,:,:]
        r_bare_gd = (jnp.einsum('silb,slja->sijba', r_bare, vel) - jnp.einsum('sila,sljb->sijba', vel, r_bare) - epsilon_gd[:,:,:,:, None] * r_bare[:,:,:, None,:]) / epsilon[:,:,:, None, None]
        r_bare_gd=jnp.where(jnp.isinf(r_bare_gd), 0.,r_bare_gd)
        self.r_bare=r_bare
        self.r_bare_gd=r_bare_gd[:, nf - nv:nf + nc, nf - nv:nf + nc,:,:]
        self.epsilon=epsilon[:, nf - nv:nf + nc, nf - nv:nf + nc]
        self.epsilon_gd=epsilon_gd[:, nf - nv:nf + nc, nf - nv:nf + nc,:]
        print("r_bare solved")
    
    @partial(jit, static_argnums=(0, 1))
    def mk_t(self, Omega):
        num_d_basis = self.num_d_basis
        nv, nc, nf = self.nv, self.nc, self.nf
        eta = self.eta
        r_bare = self.r_bare
        Omega_p, Omega_m = Omega + 1j * eta, Omega - 1j * eta
        Omegas = jnp.asarray([Omega_p, -Omega_m, 2 * Omega_m, -2 * Omega_p]).flatten()
        W = self.W_val_grad[:,:, nf - nv:nf + nc, nf - nv:nf + nv,:,:, 0]
        H_eh = jnp.diag(self.epsilon[:, nv:nv + nc, 0:nv].flatten()) - jnp.transpose(W, axes=(0, 2, 3, 1, 4, 5))[:, nv:nv + nc, 0:nv,:, nv:nv + nc, 0:nv].reshape((num_d_basis * nc * nv, num_d_basis * nc * nv))
        K = jnp.transpose(W, axes=(0, 2, 3, 1, 5, 4))[:, nv:nv + nc, 0:nv,:, nv:nv + nc, 0:nv].reshape((num_d_basis * nc * nv, num_d_basis * nc * nv))
        r_cv_bare = r_bare[:, nf:nf + nc, nf - nv:nf,:].reshape((num_d_basis * nc * nv, self.dim))
        @partial(jit,static_argnums=(1,2,3))
        def h_cv(omega, H_eh, K, r_cv_bare):
            l = len(H_eh)
            U = K @ jnp.linalg.inv(omega * jnp.eye(l) + H_eh.conj())
            r_vc_bare = r_cv_bare.conj()
            y = r_cv_bare + U @ r_vc_bare
            A = omega * jnp.eye(l) - H_eh + U @ (K.conj())
            return jnp.linalg.solve(A, y)
        h_cv = vmap(h_cv, (0, None, None, None))
        h_cv_p, h_cv_m, h2_cv_p, h2_cv_m = h_cv(Omegas, H_eh, K, r_cv_bare).reshape((4, len(Omega), num_d_basis * nc * nv, self.dim))

        W_val_grad = jnp.transpose(self.W_val_grad, axes=(0, 2, 3, 1, 4, 5, 6))[:,:,:,:, nv:nv + nc, 0:nv,:].reshape((num_d_basis, self.num_nd_bands, self.num_nd_bands, num_d_basis * nc * nv, self.dim + 1))
        W = W_val_grad[:,:,:,:, 0]
        W_grad = W_val_grad[:,:,:,:, 1:]
        
        f = self.vec2arr(jnp.where(jnp.arange(self.num_nd_bands) < nf, 1, 0))
        t_p= -1.* jnp.einsum('sijm,oma->soija', W, h_cv_p)
        t_E_p=t_p*jnp.abs(f)[None,None,:,:,None]
        t_p_gd = 1j * (jnp.einsum('silb,solja->soijba', r_bare, t_p) - jnp.einsum('soila,sljb->soijba', t_p, r_bare)) - jnp.einsum('sijmb,oma->soijba', W_grad, h_cv_p)
        t_E_p_gd=1j * (jnp.einsum('silb,solja->soijba', r_bare, t_E_p) - jnp.einsum('soila,sljb->soijba', t_E_p, r_bare)) - jnp.einsum('sijmb,oma->soijba', W_grad, h_cv_p)*jnp.abs(f)[None,None,:,:,None,None]
        t_m= -1.* jnp.einsum('sijm,oma->soija', W, h_cv_m)
        t_E_m=t_m * jnp.abs(f)[None,None,:,:, None]
        t_m_gd=1j * (jnp.einsum('silb,solja->soijba', r_bare, t_m) - jnp.einsum('soila,sljb->soijba', t_m, r_bare)) - jnp.einsum('sijmb,oma->soijba', W_grad, h_cv_m)
        t_E_m_gd=1j * (jnp.einsum('silb,solja->soijba', r_bare, t_E_m) - jnp.einsum('soila,sljb->soijba', t_E_m, r_bare)) - jnp.einsum('sijmb,oma->soijba', W_grad, h_cv_m)*jnp.abs(f)[None,None,:,:,None,None]
        t = t_p + jnp.transpose(t_m, axes=(0, 1, 3, 2, 4)).conj()
        t_E = t_E_p + jnp.transpose(t_E_m, axes=(0, 1, 3, 2, 4)).conj()
        t_gd = t_p_gd + jnp.transpose(t_m_gd, axes=(0, 1, 3, 2, 4, 5)).conj()
        t_E_gd = t_E_p_gd + jnp.transpose(t_E_m_gd, axes=(0, 1, 3, 2, 4, 5)).conj()

        t2_p= -1.* jnp.einsum('sijm,oma->soija', W, h2_cv_p)
        t2_m= -1.* jnp.einsum('sijm,oma->soija', W, h2_cv_m)
        t2 = t2_m + jnp.transpose(t2_p, axes=(0, 1, 3, 2, 4)).conj()
        t2_E=t2*jnp.abs(f)[None,None,:,:,None]

        self.t = t[:,:, nf - nv:nf + nc, nf - nv:nf + nc,:]
        self.t_gd = t_gd[:,:, nf - nv:nf + nc, nf - nv:nf + nc,:,:]
        self.t_E = t_E[:,:, nf - nv:nf + nc, nf - nv:nf + nc,:]
        self.t_E_gd = t_E_gd[:,:, nf - nv:nf + nc, nf - nv:nf + nc,:,:]
        self.t2 = t2[:,:, nf - nv:nf + nc, nf - nv:nf + nc,:]
        self.t2_E = t2_E[:,:, nf - nv:nf + nc, nf - nv:nf + nc,:]
        print("t solved")

    def SHG(self, Omega, nv, nf, nc, eta=0.05):
        self.eta = eta
        self.nv = nv
        self.nf = nf
        self.nc = nc
        Omega_p = Omega + 1j * eta
        self.G_list = list(set(map(lambda nd_qtm: self.separate(nd_qtm, self.dim)[0], self.nd_basis)))
        self.G_array = np.array(self.G_list)
        self.coarse_k_list = list(set(map(lambda d_qtm: self.separate(d_qtm, self.dim)[0], self.d_qtm_list)))
        self.d_qtm_nk_list = list(set(map(lambda d_qtm: self.separate(d_qtm, self.dim)[1], self.d_qtm_list)))
        self.interaction_grad = jacfwd(self.interaction, argnums=(0))
        self.vel_opt = jacfwd(self.hm,argnums=(0))
        self.mk_hamiltonian()
        self.solve()
        self.mk_overlape_idx()
        print("overlape index make")
        self.mk_W()
        self.mk_r_bare()
        self.mk_t(Omega)
        self.r_bare = self.r_bare[:, nf - nv:nf + nc, nf - nv:nf + nc,:]

        f = self.vec2arr(jnp.array([1 for i in range(nv)] + [0 for i in range(nc)]))

        @jit
        def shg(r, r_gd, r2, epsilon, epsilon_gd, Omega, f):
            h = r / (Omega[:, None, None, None] - epsilon[None,:,:, None])
            g = h * (f.T)[None,:,:, None]
            g_gd = (r_gd * (f.T)[None,:,:, None, None] + epsilon_gd[None,:,:,:, None] * g[:,:,:, None,:]) / (Omega[:, None, None, None, None] - epsilon[None,:,:, None, None])
            h2 = r2 / (-2.*Omega[:, None, None, None] - epsilon[None,:,:, None])
            res = (jnp.transpose(h2, axes=(0, 2, 1, 3))[:,:,:,:, None, None]) * ((jnp.einsum('oilb,olja->oijba', r, g) - jnp.einsum('oila,oljb->oijba', g, r) + 1j * g_gd)[:,:,:, None,:,:]) + 1j / 2.*jnp.transpose(h,axes=(0,2,1,3))[:,:,:,None,:,None]*g_gd[:,:,:,:,None,:]
            return -jnp.sum(res, axis=(1, 2))
        
        shg = vmap(shg, (0, 0, 0, 0, 0, None, None))
        r_bare=self.r_bare[:, None,:,:,:]
        r_bare_gd = self.r_bare_gd[:,None,:,:,:,:]
        ipa = jnp.sum(shg(r_bare,r_bare_gd,r_bare,self.epsilon,self.epsilon_gd,Omega_p,f), axis = 0)
        exc = jnp.sum(shg(r_bare + self.t_E, r_bare_gd + self.t_E_gd, r_bare + self.t2_E, self.epsilon, self.epsilon_gd, Omega_p, f), axis = 0)
        exc_aug = jnp.sum(shg(r_bare + self.t, r_bare_gd + self.t_gd, r_bare + self.t2, self.epsilon, self.epsilon_gd, Omega_p, f) , axis = 0)
                

        plt.plot(Omega, jnp.abs(ipa[:, 0, 0, 0]) / self.num_d_basis * Omega)
        plt.plot(Omega, jnp.abs(exc[:, 0, 0, 0]) / self.num_d_basis * Omega)
        plt.plot(Omega, jnp.abs(exc_aug[:, 0, 0, 0]) / self.num_d_basis * Omega)
        plt.savefig("temp_shg_16.pdf")
        #plt.show()

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
