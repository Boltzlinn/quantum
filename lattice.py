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
    def mk_interaction_tensor(self):
        delta_d_qtm_nk_list = list(set([self.qtm_minus(d_qtm_nk1, d_qtm_nk2) for d_qtm_nk1 in self.d_qtm_nk_list for d_qtm_nk2 in self.d_qtm_nk_list]))
        print(len(delta_d_qtm_nk_list))
        q_array = (np.array(self.fine_k_list)[:, None, None,:] - np.array(self.coarse_k_list)[None,:, None,:] + np.array(self.G_list)[None, None,:,:]).reshape((-1, 2))
        print(len(q_array))
        q_list = list(set([tuple(q) for q in q_array]))
        q_array = np.asarray(q_list)
        self.q_list = {q_list[i]:i for i in range(len(q_list))}
        print(len(q_list))
        interaction_tensor = {}
        Inter = self.interaction
        Inter_grad = jacfwd(self.interaction,argnums=(0))
        
        for delta_d_qtm_nk in delta_d_qtm_nk_list:
            interaction = np.array(vmap(partial(Inter, delta_d_qtm_nk=delta_d_qtm_nk))(q_array))
            interaction_grad = np.array(vmap(partial(Inter_grad, delta_d_qtm_nk=delta_d_qtm_nk))(q_array))
            interaction_tensor[delta_d_qtm_nk] = np.hstack((interaction.reshape((-1,1)), interaction_grad))
        self.interaction_tensor = interaction_tensor

    @staticmethod
    @jit
    def overlape(wv_func_1, wv_func_2, idxs):
        idx1, idx2 = idxs
        wv1 = wv_func_1[idx1]
        wv2 = wv_func_2[idx2]
        return jnp.einsum('ji,jk->ik',wv1.conj(),wv2)

    @partial(jit, static_argnums=(0))
    def hartree(self, delta_d_qtm_nk, wv_func_1, wv_func_2):
        #idx = list(itemgetter(*self.G_list)(self.q_list))
        #interaction = self.interaction_tensor[delta_d_qtm_nk][idx, 0]
        interaction = vmap(partial(self.interaction, delta_d_qtm_nk=delta_d_qtm_nk))(np.array(self.G_list))
        overlape_1 = jnp.einsum('ei,gef,fj->gij', wv_func_1.conj(), self.overlape_idx, wv_func_1)
        overlape_2 = jnp.einsum('ei,gef,fj->gij', wv_func_2.conj(), self.overlape_idx, wv_func_2)
        res = jnp.einsum('g,gij,gkl->ijkl',interaction, overlape_1, overlape_2.conj())
        return res
    
    @partial(jit, static_argnums=(0))
    def fock(self, delta_k, delta_d_qtm_nk, wv_func_1, wv_func_2):
        q_array = delta_k + np.array(self.G_list)
        #q_list = list(set([tuple(q) for q in q_array]))
        #idx = list(itemgetter(*q_list)(self.q_list))
        #interaction_val_grad = self.interaction_tensor[delta_d_qtm_nk][idx]
        interaction = vmap(partial(self.interaction, delta_d_qtm_nk=delta_d_qtm_nk))(q_array)
        interaction_grad = vmap(partial(self.interaction_grad, delta_d_qtm_nk=delta_d_qtm_nk))(q_array)
        interaction_val_grad = jnp.hstack((interaction.reshape((-1,1)), interaction_grad))
        overlape = jnp.einsum('ei,gef,fj->gij', wv_func_1.conj(), self.overlape_idx, wv_func_2)
        res_val_grad = jnp.einsum('ga,gik,gjl->ijkla', interaction_val_grad, overlape, overlape.conj())
        return res_val_grad

    @partial(jit, static_argnums=(0))
    def W(self, delta_k, delta_d_qtm_nk, wv_func_1, wv_func_2):
        hartree = self.hartree(delta_d_qtm_nk, wv_func_1, wv_func_2)
        fock_val_grad = self.fock(jnp.array(delta_k), delta_d_qtm_nk, wv_func_1, wv_func_2)
        return fock_val_grad - (hartree[:,:,:,:,None])*(np.array([1.]+[0. for i in range(self.dim)])[None,None,None,None,:])

    def testW(self):
        self.mk_hamiltonian()
        self.solve()
        self.mk_overlape_idx()
        d_qtm1 = list(self.Basis.keys())[0]
        d_qtm2 = list(self.Basis.keys())[1]
        W = self.W(d_qtm1, d_qtm2, self.wv_funcs[d_qtm1], self.wv_funcs[d_qtm2])[0]
        print(jnp.max(jnp.abs(W-jnp.transpose(W,(1,0,3,2)).conj())))

    #@partial(jit, static_argnums=(0, 1, 2, 3))
    def bse_solve(self, nv, nf, nc):
        num_d_basis = self.num_d_basis
        '''
        wv_funcs = jnp.transpose(jnp.array([self.wv_funcs[d_qtm][:, nf - nv:nf + nc] for d_qtm in self.Basis]), axes=(1, 2, 0)).reshape((self.num_nd_bands, (nc + nv) * self.num_d_basis))
        print("wave get")
        I = jnp.array([self.overlape(wv_funcs, wv_funcs, self.overlape_idx[G]) for G in self.G_list]).reshape((len(self.G_list), nc + nv, self.num_d_basis, nc + nv, self.num_d_basis))  #[G,i,k1,j,k2]
        print("overlape get")
        hartree = jnp.array([[[self.interaction_tensor[self.separate(self.qtm_minus(d_qtm1, d_qtm2), self.dim)[1]][G][0] for d_qtm2 in self.Basis] for d_qtm1 in self.Basis] for G in self.G_list])
        print("hartree get")
        fock = jnp.array([[[self.interaction_tensor[self.separate(self.qtm_minus(d_qtm1, d_qtm2), self.dim)[1]][self.qtm_add(self.separate(self.qtm_minus(d_qtm1, d_qtm2), self.dim)[0], G)][0] for d_qtm2 in self.Basis] for d_qtm1 in self.Basis] for G in self.G_list])
        print("fock get")
        fock_grad = jnp.array([[[self.interaction_tensor[self.separate(self.qtm_minus(d_qtm1, d_qtm2), self.dim)[1]][self.qtm_add(self.separate(self.qtm_minus(d_qtm1, d_qtm2), self.dim)[0], G)][1] for d_qtm2 in self.Basis] for d_qtm1 in self.Basis] for G in self.G_list])
        print("fock grad get")
        W = jnp.einsum('gst,giskt,gjslt->stijkl', fock, I, I.conj()) - jnp.einsum('gss,gisjs,gktlt->stijkl', hartree, I, I.conj())
        print("W get")
        W_tangents = jnp.einsum('gsta,giskt,gjslt->stijkla', fock_grad, I, I.conj())
        print("W grad get")
        '''
        W_val_grad = jnp.array([[self.W(jnp.array(d_qtm1)[0:self.dim]-jnp.array(d_qtm2)[0:self.dim], self.separate(self.qtm_minus(d_qtm1, d_qtm2), self.dim)[1], self.wv_funcs[d_qtm1][:, nf - nv:nf + nc], self.wv_funcs[d_qtm2][:, nf - nv:nf + nc]) for d_qtm2 in self.Basis] for d_qtm1 in self.Basis])  #[num_d,num_d,nv+nv,nv+nv,nv+nv,nv+nv,dim+1]
        print(jnp.shape(W_val_grad))
        W = W_val_grad[:,:,:,:,:,:, 0]
        W_grad = W_val_grad[:,:,:,:,:,:, 1:]
        self.w = W
        self.w_grad = W_grad
        print("W calculated")
        H_ep_0 = jnp.array([self.vec2arr(self.energies[d_qtm]) for d_qtm in self.Basis])  #[num_d,nv+nv,nv+nv]
        H_ep = jnp.diag(H_ep_0[:, nv:nv + nc, 0:nv].flatten()) - jnp.transpose(W, axes=(0, 2, 3, 1, 4, 5))[:, nv:nv + nc, 0:nv,:, nv:nv + nc, 0:nv].reshape((num_d_basis * nc * nv, num_d_basis * nc * nv))
        K = jnp.transpose(W, axes=(0, 2, 3, 1, 5, 4))[:, nv:nv + nc, 0:nv,:, nv:nv + nc, 0:nv].reshape((num_d_basis * nc * nv, num_d_basis * nc * nv))
        self.H_ep = H_ep
        self.K = K
        exciton_energies, exciton_wv_funcs = jnp.linalg.eigh(H_ep)
        print("eigen solved")
        self.exciton_energies = exciton_energies
        self.exciton_wv_funcs = exciton_wv_funcs
        r_cv_bare = []
        i=0
        for d_qtm in self.Basis:
            k, d_qtm_nk = self.separate(d_qtm, self.dim)
            k = jnp.array(k)
            vel = jnp.einsum('lk,lma,mn->kna', self.wv_funcs[d_qtm].conj()[:, nf:nf + nc], self.vel_opt(k, d_qtm_nk), self.wv_funcs[d_qtm][:, nf - nv:nf])
            r_cv_bare = r_cv_bare + [-1j * vel / H_ep_0[i][nf:nf + nc, nf - nv:nf, None]]
            
            i=i+1
        self.r_cv_bare = jnp.array(r_cv_bare).reshape((num_d_basis * nc * nv, self.dim))
        print(jnp.max(jnp.where(jnp.isnan(self.r_cv_bare), 1, 0)))
        print("r_cv solved")
    
    #@partial(jit, static_argnums=(0))
    def h_cv(self, omega):
        G = (self.exciton_wv_funcs.conj() * ((1 / (omega + self.exciton_energies))[None,:])) @ self.exciton_wv_funcs.T
        KG = self.K @ G
        r_vc_bare = (self.r_cv_bare).conj()
        y = self.r_cv_bare + KG @ r_vc_bare
        A = omega * jnp.eye(len(G)) - self.H_ep + KG @ (self.K.conj())
        Am = jnp.max(A)
        ym = jnp.max(y)
        h_cv = jnp.linalg.solve(A, y)
        return h_cv
    
    #@partial(jit, static_argnums=(0, 1))
    def mk_h_cv(self, Omega):
        eta = 0.1
        h_cv_o = vmap(self.h_cv)(Omega + 1j * eta)
        h_cv_mo = vmap(self.h_cv)(-1.*Omega + 1j * eta)
        h_cv_2o = vmap(self.h_cv)(2.*Omega - 2*1j * eta)
        h_cv_2mo = vmap(self.h_cv)(-2.*Omega - 2 * 1j * eta)
        self.h_cv_o = h_cv_o
        self.h_cv_mo = h_cv_mo
        self.h_cv_2o = h_cv_2o
        self.h_cv_2mo = h_cv_2mo
        print("h_cv_bar solved")

    @partial(jit, static_argnums=(0, 2, 3, 4, 5, 6))
    def shg(self, k, d_qtm_nk, Omega, nv, nf, nc):
        eta = 0.1
        energies, wv_func = jnp.linalg.eigh(self.hm(k, d_qtm_nk))
        vel = jnp.einsum('lk,lma,mn->kna', wv_func.conj(), self.vel_opt(k, d_qtm_nk), wv_func)  #[nv+nc,nv+nc,dim]
        
        f = self.vec2arr(jnp.array([1 for i in range(nv)] + [0 for i in range(nc)]))  #[nv+nc,nv+nc]
        epsilon = self.vec2arr(energies)  #[nv+nc,nv+nc]

        r_bare = -1j * vel/epsilon[:,:,None]  #[nv+nc,nv+nc,dim]
        r_bare = jnp.where(jnp.isinf(r_bare), 0.,r_bare)  #[nv+nc,nv+nc,dim]

        delta=vmap(self.vec2arr, 1, 2)(vmap(jnp.diag, 2, 1)(vel))  #[nv+nc,nv+nc,dim]
        
        r_bare_gd = (jnp.einsum('ilb,lja->ijba', r_bare, vel) - jnp.einsum('ila,ljb->ijba', vel, r_bare) - delta[:,:,:, None] * r_bare[:,:, None,:]) / epsilon[:,:, None, None]
        r_bare_gd=jnp.where(jnp.isinf(r_bare_gd), 0.,r_bare_gd)
        
        vel=vel[nf - nv:nf + nc, nf - nv:nf + nc,:]
        epsilon=epsilon[nf - nv:nf + nc, nf - nv:nf + nc]
        r_bare=r_bare[nf - nv:nf + nc, nf - nv:nf + nc,:]
        r_bare_gd=r_bare_gd[nf - nv:nf + nc, nf - nv:nf + nc,:,:]
        delta = delta[nf - nv:nf + nc, nf - nv:nf + nc,:]
        
        W_val_grad=jnp.array([self.W(k - jnp.array(self.separate(d_qtm_bar, self.dim)[0]), self.qtm_minus(d_qtm_nk, self.separate(d_qtm_bar, self.dim)[1]), wv_func[:, nf - nv:nf + nc], self.wv_funcs[d_qtm_bar][:, nf - nv:nf + nc]) for d_qtm_bar in self.Basis])
        W = W_val_grad[:,:,:,:,:, 0]
        W_grad = W_val_grad[:,:,:,:,:, 1:]
        '''
        i=self.Basis[d_qtm][0]
        W=self.w[i]
        W_tangents = self.w_grad[i]
        '''
        W=jnp.transpose(W, axes=(1, 2, 0, 3, 4))[:,:,:, nv:nv + nc, 0:nv].reshape((nv + nc, nv + nc, self.num_d_basis * nc * nv))
        W_grad=jnp.transpose(W_grad, axes=(1, 2, 0, 3, 4, 5))[:,:,:, nv:nv + nc, 0:nv,:].reshape((nv + nc, nv + nc, self.num_d_basis * nc * nv, self.dim))
        
        
        t_o = -1.* jnp.einsum('ijm,oma->oija', W, self.h_cv_o)
        t_o_gd=1j * (jnp.einsum('ilb,olja->oijba', r_bare, t_o) - jnp.einsum('oila,ljb->oijba', t_o, r_bare)) - jnp.einsum('ijmb,oma->oijba', W_grad, self.h_cv_o)
        
        t_mo = -1.* jnp.einsum('ijm,oma->oija', W, self.h_cv_mo)
        t_mo_gd=1j * (jnp.einsum('ilb,olja->oijba', r_bare, t_mo) - jnp.einsum('oila,ljb->oijba', t_mo, r_bare)) - jnp.einsum('ijmb,oma->oijba', W_grad, self.h_cv_mo)
        
        t=t_o + jnp.transpose(t_mo, axes=(0, 2, 1, 3)).conj()
        t_gd=t_o_gd + jnp.transpose(t_mo_gd, axes=(0, 2, 1, 3, 4)).conj()
        
        r=r_bare + t  #[num_o,nc+nv,nc+nv,dim]
        r_gd=r_bare_gd + t_gd  #[num_o,nc+nv,nc+nv,dim]
        
        h_o=r / (Omega[:, None, None, None] + 1j * eta - epsilon[None,:,:, None])
        g=h_o * (f.T)[None,:,:, None]
        g_gd=(r_gd * (f.T)[None,:,:, None, None] + delta[None,:,:,:, None] * g[:,:,:, None,:]) / (Omega[:, None, None, None, None] + 1j * eta - epsilon[None,:,:, None, None])
        
        t_2o= -1.* jnp.einsum('ijm,oma->oija', W, self.h_cv_2o)
        t_2mo= -1.* jnp.einsum('ijm,oma->oija', W, self.h_cv_2mo)
        t_2=t_2mo + jnp.transpose(t_2o, axes=(0, 2, 1, 3)).conj()
        r_2=r_bare + t_2
        h_2mo=r_2 / (-2.*Omega[:, None, None, None] - 2.* 1j * eta - epsilon[None,:,:, None])
        
        res=(jnp.transpose(h_2mo, axes=(0, 2, 1, 3))[:,:,:,:, None, None]) * ((jnp.einsum('oilb,olja->oijba', r, g) - jnp.einsum('oila,oljb->oijba', g, r) + 1j * g_gd)[:,:,:, None,:,:]) + 1j / 2.*jnp.transpose(h_o,axes=(0,2,1,3))[:,:,:,None,:,None]*g_gd[:,:,:,:,None,:]
        
        return -jnp.sum(res, axis=(1, 2))

    def SHG(self, Omega, nv, nf, nc, k_mesh_fined_multiples = 1):
        num_O = len(Omega)
        self.G_list = list(set(map(lambda nd_qtm: self.separate(nd_qtm, self.dim)[0], self.nd_basis)))
        d_qtm_list = list(self.Basis.keys())
        self.coarse_k_list = list(set(map(lambda d_qtm: self.separate(d_qtm, self.dim)[0], d_qtm_list)))
        self.d_qtm_nk_list = list(set(map(lambda d_qtm: self.separate(d_qtm, self.dim)[1], d_qtm_list)))
        k_mesh_fined_multiples = tuple(np.ones(self.dim, dtype=np.int32) * k_mesh_fined_multiples)
        if k_mesh_fined_multiples == tuple((1 for i in range(self.dim))):
            self.fine_k_list = self.coarse_k_list
        else:
            fine_k_list = [[]]
            for i in range(self.dim):
                fine_k_list = [fine_k + [self.kstart[i] + n / self.real_shape[i] / k_mesh_fined_multiples[i]] for fine_k in fine_k_list for n in range(self.real_shape[i] * k_mesh_fined_multiples[i])]
            self.fine_k_list = fine_k_list
        self.interaction_grad = jacfwd(self.interaction, argnums=(0))
        self.vel_opt = jacfwd(self.hm,argnums=(0))
        self.mk_hamiltonian()
        self.solve()
        self.mk_overlape_idx()
        print("overlape index make")
        #self.mk_interaction_tensor()
        #print("interaction tensor make")
        self.bse_solve(nv, nf, nc)
        self.mk_h_cv(Omega)
        
        res = np.zeros((num_O, self.dim, self.dim, self.dim), dtype=np.complex128)
        flag = 0
        for d_qtm_nk in self.d_qtm_nk_list:
            for k in self.fine_k_list:
                print(flag)
                flag = flag + 1
                k = jnp.array(k)
                resp = self.shg(k, d_qtm_nk, Omega, nv, nf, nc)
                res = res + np.array(resp)
                

        plt.plot(Omega, jnp.abs(res[:,0,0,0])/flag)
        plt.savefig("temp_shg_exc_aug.pdf")
        plt.show()
    

    @partial(jit, static_argnums=(0,2,3,4))
    def shg_ipa(self, k, d_qtm_nk, Omega, bd_range):
        eta = 0.03
        bot, top = bd_range
        eigval, wvfunc = jnp.linalg.eigh(self.hm(k, d_qtm_nk))
        vel = jnp.einsum('lk,lma,mn->kna', wvfunc.conj(), self.vel_opt(k, d_qtm_nk), wvfunc)  #[bd_range,bd_range,dim]

        dist = jnp.where(eigval > 0., 0, 1)
        d_dist = self.vec2arr(dist)  #[bd_range,bd_range]
        d_en = self.vec2arr(eigval)  #[bd_range,bd_range]

        r_inter = -1j * vel/d_en[:,:,None]  #[bd_range,bd_range,dim]
        r_inter = jnp.where(jnp.isinf(r_inter), 0.,r_inter)  #[bd_range,bd_range,dim]

        d_en_jac = vmap(self.vec2arr, 1, 2)(vmap(jnp.diag,2,1)(vel))  #[bd_range,bd_range,dim]
        
        r_inter_gd = (r_inter[:,:,:, None] * jnp.transpose(d_en_jac, axes=(1, 0, 2))[:,:, None,:] + r_inter[:,:, None,:] * (jnp.transpose(d_en_jac, axes=(1, 0, 2))[:,:,:, None]) + 1j * jnp.einsum('nla,lmb->nmab', r_inter, r_inter * d_en[:,:, None]) - 1j * jnp.einsum('nlb,lma->nmab', r_inter * d_en[:,:, None], r_inter)) / d_en[:,:, None, None]
        r_inter_gd=jnp.where(jnp.isinf(r_inter_gd), 0.,r_inter_gd)  #[bd_range,bd_range,dim,dim]
        
        r_inter=r_inter[bot:top, bot:top,:]
        r_inter_gd=r_inter_gd[bot:top, bot:top,:,:]
        d_en=d_en[bot:top, bot:top]
        d_en_jac = d_en_jac[bot:top,bot:top,:]
        d_dist = d_dist[bot:top,bot:top]

        f = r_inter[None,:,:,:] / (Omega[:,None,None,None] + 1j * eta - d_en[None,:,:,None])  #[nOmega,bd_range,bd_range,dim]
        h = r_inter[None,:,:,:] / (2 * Omega[:, None, None, None] + 2 * 1j * eta + d_en[None,:,:, None])  #[nOmega,bd_range,bd_range,dim]
        
        g = d_dist.T[None,:,:, None] * f  #[nOmega,bd_range,bd_range,dim]
        g_gd = (d_dist.T[None,:,:, None, None] * r_inter_gd[None,:,:,:,:] + g[:,:,:, None,:] * d_en_jac[None,:,:,:, None]) / (Omega[:, None, None, None, None] + 1j * eta - d_en[None,:,:, None, None])  #[nOmega,bd_range,bd_range,dim,dim]
        
        rg = 1j * g_gd + jnp.einsum('nlb,olma->onmba', r_inter, g) - jnp.einsum('onla,lmb->onmba', g, r_inter)  #[nOmega,bd_range,bd_range,dim,dim]
        res = jnp.transpose(h, (0, 2, 1, 3))[:,:,:,:, None, None] * rg[:,:,:, None,:,:] - 1j / 2 * jnp.transpose(f, (0, 2, 1, 3))[:,:,:, None, None,:] * g_gd[:,:,:,:,:, None]  #[nOmega,bd_range,bd_range,dim,dim,dim]
        return jnp.sum(res, axis=(1, 2))

    

    def SHG_ipa(self, Omega, bd_range):
        dim = self.dim
        num_O = len(Omega)
        self.vel_opt = jacfwd(self.hm,argnums=(0))
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
