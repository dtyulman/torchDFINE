import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt

from plot_utils import prep_axes


class NonlinearStateSpaceModel():
    """
    Dynamic latent:  x_{t+1} = A(x_t) + B*u_t + q_t,  q_t ~ N(0,Q)
    Manifold latent:     a_t = C*x_t + r_t,           w_t ~ N(0,R)
    Observation:         y_t = f(a_t) + s_t,          r_t ~ N(0,S)
    """
    def __init__(self, A_fn,B,C,f, Q,R,S, noise=True):
        #convenience variables
        self.dim_a, self.dim_x = C.shape
        _, self.dim_u = B.shape
        self.dim_y = f(torch.zeros(self.dim_a)).shape[-1] #dummy input to f

        #sanity check dimensions
        assert self.dim_x == A_fn.A.shape[0] == A_fn.A.shape[1] == B.shape[0] == C.shape[1] == Q.shape[0] == Q.shape[1]
        assert self.dim_a == C.shape[1] == R.shape[0] == R.shape[1]
        assert self.dim_y == S.shape[0] == S.shape[1]

        #transition/observation functions
        self.A_fn = A_fn #(x->x)
        self.B = B #[x,u]
        self.C = C #[a,x]
        self.f = f #(y->a)

        #noise distributions
        self.Q_distr = MultivariateNormal(torch.zeros(self.dim_x), Q) #[x,x]
        self.R_distr = MultivariateNormal(torch.zeros(self.dim_a), R) #[a,a]
        self.S_distr = MultivariateNormal(torch.zeros(self.dim_y), S) #[y,y]

        #if False, removes all noise from the system
        self.global_noise_toggle = noise


    def __repr__(self):
        return (f'{self.A_fn}\n'
                f'B={self.B.numpy()}\n'
                f'Q={self.Q_distr.covariance_matrix.numpy()}\n'
                '--\n'
                f'C={self.C.numpy()}\n'
                f'R={self.R_distr.covariance_matrix.numpy()}\n'
                '--\n'
                f'{self.f}\n'
                f'S={self.S_distr.covariance_matrix.numpy()}\n')


    def step_dynamic_latent(self, x, u=None):
        x_next = self.A_fn( x )
        if u is not None:
            x_next += self.B @ u
        if self.global_noise_toggle:
            x_next += self.Q_distr.sample()
        return x_next


    def compute_manifold_latent(self, x):
        a = self.C @ x
        if self.global_noise_toggle:
            a += self.R_distr.sample()
        return a


    def compute_observation(self, a, noise=True):
        y = self.f( a )
        if self.global_noise_toggle and noise:
            y += self.S_distr.sample()
        return y


    def step(self, x, u=None):
        x_next = self.step_dynamic_latent(x, u)
        a_next = self.compute_manifold_latent(x_next)
        y_next = self.compute_observation(a_next)
        return x_next, a_next, y_next


    def generate_trajectory(self, x0, u_seq=None, num_steps=None, num_seqs=None):
        if num_steps is None:
            if len(u_seq.shape) == 2: #shape is [t,u]
                num_steps = u_seq.shape[0]
            elif len(u_seq.shape) == 3: #shape is [b,t,u]
                num_steps = u_seq.shape[1]

        _num_seqs = 1 if num_seqs is None else num_seqs

        x_seq = torch.empty(_num_seqs, num_steps, self.dim_x) #[b,t,x]
        a_seq = torch.empty(_num_seqs, num_steps, self.dim_a) #[b,t,a]
        y_seq = torch.empty(_num_seqs, num_steps, self.dim_y) #[b,t,y]

        for b in range(_num_seqs): #TODO: vectorize
            x_seq[b,0] = x0[b]
            a_seq[b,0] = self.compute_manifold_latent(x_seq[b,0])
            y_seq[b,0] = self.compute_observation(a_seq[b,0])
            for t in range(1, num_steps):
                x_seq[b,t], a_seq[b,t], y_seq[b,t] = self.step(x_seq[b,t-1], u_seq[b,t-1])

        if num_seqs is None:
            x_seq, a_seq, y_seq = x_seq.squeeze(0), a_seq.squeeze(0), y_seq.squeeze(0)
        return x_seq, a_seq, y_seq


#%%
class AffineTransformation():
    def __init__(self, A, b=None):
        self.A = A
        self.b = b if b is not None else 0

    def __call__(self, x):
        return self.A @ x + self.b

    def __repr__(self):
        A = self.A.detach().numpy() if torch.is_tensor(self.A) else self.A
        b = self.b.detach().numpy() if torch.is_tensor(self.b) else self.b
        return f'A={A}\nb={b}'


#%%
class NonlinearEmbeddingBase():
    """
    y = T * e(a)
    """
    nonlin_dim = None # Must be set in subclass

    def __init__(self, output_dim=None, lo=-5, hi=5):
        self.output_dim = self.nonlin_dim if output_dim is None else output_dim
        assert self.output_dim >= self.nonlin_dim

        proj_eye = torch.eye(self.nonlin_dim) #first 3 dimensions are identical to embedding, for vizualization
        proj_rand = (hi-lo) * torch.rand(self.output_dim-self.nonlin_dim, self.nonlin_dim) + lo #the other dims are random
        self.projection_matrix = torch.vstack([proj_eye, proj_rand]) #high-dimensional projection matrix


    def __call__(self, d):
        #input is [b,t,a] or [t,a] or [a]
        #output is  [b,t,y] or [t,y] or [y]
        return (self.projection_matrix @ self.nonlin_embed(d).unsqueeze(-1)).squeeze(-1)


    def nonlin_embed(self, *args):
        raise NotImplementedError('Override this method')


    def inv(self, y):
        #input  is [b,t,y] or [t,y] or [y]
        #output is [b,t,n] or [t,n] or [n]
        e = torch.pinverse(self.projection_matrix) @ y.mT
        d = self.nonlin_embed_inv(e)
        return d


    def nonlin_embed_inv(self, e):
        raise NotImplementedError('Override this method')


    def _plot_manifold(self, *args, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        else:
            fig = ax.get_figure()

        y = self.nonlin_embed(*args)
        ax.scatter(*y.T, color='grey', alpha=0.2)
        return fig, ax


    def plot_manifold(self):
        raise NotImplementedError('Override this method')



class RingManifold(NonlinearEmbeddingBase):
    nonlin_dim = 3

    def nonlin_embed(self, d):
        #input  is [b,t,1] or [t,1] or [1]
        #output is [b,t,3] or [t,3] or [3]
        e = torch.stack([torch.cos(d),
                         torch.sin(2*d),
                         torch.sin(d)], dim=-1)
        return e


    def __repr__(self):
        return (f'T={self.projection_matrix.numpy()}\n'
                f'e=[cos(a); sin(2a); sin(a)]')


    def plot_manifold(self, ax=None):
        d = torch.linspace(0, 2*torch.pi, 100).unsqueeze(-1)
        return self._plot_manifold(d, ax=ax)



class SwissRoll(NonlinearEmbeddingBase):
    nonlin_dim = 3

    def nonlin_embed(self, d):
        #input  is [b,t,2] or [t,2] or [2]
        #output is [b,t,3] or [t,3] or [3]
        r, h = d[..., 0], d[..., 1] #radius, height
        e = torch.stack([0.5*r*torch.cos(r),
                         h,
                         0.5*r*torch.sin(r)], dim=-1)
        return e


    def __repr__(self):
        return (f'T={self.projection_matrix.numpy()}\n'
                f'e=[r/2*cos(r); h; r/2*sin(r)]')


    def plot_manifold(self, rlim=(0, 6*torch.pi), hlim=(-1,1), ax=None):
        r = torch.linspace(rlim[0], rlim[1], 300)
        h = torch.linspace(hlim[0], hlim[1], 30)

        r,h = torch.meshgrid(r, h)
        r,h = r.flatten().unsqueeze(-1), h.flatten().unsqueeze(-1)

        d = torch.stack([r,h], dim=-1)
        return self._plot_manifold(d, ax=ax)



class IdentityManifold(NonlinearEmbeddingBase):
    def nonlin_embed(self, d):
        return d

    def nonlin_embed_inv(self, e):
        return e
