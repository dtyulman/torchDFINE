import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt

from python_utils import verify_shape


class NonlinearStateSpaceModel():
    """
    Dynamic latent:  x_{t+1} = A*x_t + b + B*u_t + q_t,  q_t ~ N(0,Q)
    Manifold latent:     a_t = C*x_t + r_t,              w_t ~ N(0,R)
    Observation:         y_t = f(a_t) + s_t,             r_t ~ N(0,S)
    """
    def __init__(self, A, b, B, C, f, Q=None, R=None, S=None):
        #check dimensions and set convenience variables
        #TODO: these should be properties so that they dont need to be updated if any of the matrices change
        self.dim_x, self.dim_u = verify_shape(B, [None, None])
        verify_shape(A, [self.dim_x, self.dim_x])
        if b is not None:
            verify_shape(b, [self.dim_x])
        else:
            b = torch.zeros(self.dim_x)
        self.dim_a, _ = verify_shape(C, [None, self.dim_x])
        self.dim_y = f(torch.zeros(self.dim_a)).shape[-1] #dummy input to f

        #transition/observation functions
        self.A = A #[x,x]
        self.b = b #[x]
        self.B = B #[x,u]
        self.C = C #[a,x]
        self.f = f #(a->y)

        #noise distributions
        self.Q_distr = self.R_distr = self.S_distr = None
        if Q is not None:
            verify_shape(Q, [self.dim_x, self.dim_x])
            self.Q_distr = MultivariateNormal(torch.zeros(self.dim_x), Q) #[x,x]
        if R is not None:
            verify_shape(R, [self.dim_a, self.dim_a])
            self.R_distr = MultivariateNormal(torch.zeros(self.dim_a), R) #[a,a]
        if S is not None:
            verify_shape(S, [self.dim_y, self.dim_y])
            self.S_distr = MultivariateNormal(torch.zeros(self.dim_y), S) #[y,y]

        #logging and state, initialized in init_state()
        self.x_seq = self.a_seq = self.y_seq = None
        self.x = self.a = self.y = self.t = None


    @staticmethod
    def init_from_DFINE(dfine):
        A = dfine.ldm.A
        b = 0
        B = dfine.ldm.B
        C = dfine.ldm.C
        f = dfine.decoder
        Q,R = dfine.ldm._get_covariance_matrices()
        return NonlinearStateSpaceModel(A, b, B, C, f, Q, R)


    def __repr__(self):
        return (f'A={self.A.numpy()}\n'
                f'b={self.b.numpy()}\n'
                f'B={self.B.numpy()}\n'
                f'Q={self.Q_distr.covariance_matrix.numpy() if self.Q_distr is not None else 0}\n'

                '--\n'

                f'C={self.C.numpy()}\n'
                f'R={self.R_distr.covariance_matrix.numpy() if self.R_distr is not None else 0}\n'

                '--\n'

                f'{self.f}\n'
                f'S={self.S_distr.covariance_matrix.numpy() if self.S_distr is not None else 0}\n')


    def next_dynamic_latent(self, x, u=None, noise=True):
        """
        inputs:
            x: [b,x]
            u: [b,u]
        returns:
            x_next: [b,x]
        """
        x_next = (self.A @ x.unsqueeze(-1)).squeeze(-1) + self.b
        if u is not None:
            x_next += (self.B @ u.unsqueeze(-1)).squeeze(-1)
        if noise and self.Q_distr is not None:
            x_next += self.Q_distr.sample(x.shape[:-1])
        return x_next


    def compute_manifold_latent(self, x, noise=True):
        """
        inputs:
            x: [b,x] or [b,t,x]
        returns:
            a: [b,a] or [b,t,a]
        """
        a = (self.C @ x.unsqueeze(-1)).squeeze(-1) #[a,x]@[..., x,1] -> [..., a,1] -> [..., a]
        if noise and self.R_distr is not None:
            a += self.R_distr.sample(a.shape[:-1])
        return a.squeeze(-1)


    def compute_observation(self, a, noise=True):
        """
        inputs:
            a: [b,a] or [b,t,a]
        returns:
            y: [b,y] or [b,t,y]
        """
        y = self.f(a)
        if noise and self.S_distr is not None:
            y += self.S_distr.sample(y.shape[:-1])
        return y


    def _update_state(self, u, noise=True):
        self.x = self.next_dynamic_latent(self.x, u, noise=noise)
        self.a = self.compute_manifold_latent(self.x, noise=noise)
        self.y = self.compute_observation(self.a, noise=noise)
        return self.x, self.a, self.y


    def _log_state(self):
        self.t += 1
        self.x_seq[:, self.t, :] = self.x
        self.a_seq[:, self.t, :] = self.a
        self.y_seq[:, self.t ,:] = self.y


    def step(self, u=None, noise=True):
        """
        inputs:
            u: [b,u]
            noise: bool
        returns:
            x_next: [b,x]
            a_next: [b,a]
            y_next: [b,y]
        """
        next_state = self._update_state(u, noise=noise)
        self._log_state()
        return next_state


    def init_state(self, x0=None, u_seq=None, num_seqs=None, num_steps=None):
        #extract/validate dimensions
        num_seqs, dim_x = verify_shape(x0, [num_seqs, self.dim_x])
        num_seqs, num_steps, dim_u = verify_shape(u_seq, [num_seqs, num_steps, self.dim_u])

        #set defaults if needed
        x0    = x0    if x0    is not None else torch.zeros(num_seqs, self.dim_x)
        u_seq = u_seq if u_seq is not None else torch.zeros(num_seqs, num_steps, self.dim_u)

        #allocate memory for logging
        self.x_seq = torch.full((num_seqs, num_steps, self.dim_x), torch.nan) #[b,t,x]
        self.a_seq = torch.full((num_seqs, num_steps, self.dim_a), torch.nan) #[b,t,a]
        self.y_seq = torch.full((num_seqs, num_steps, self.dim_y), torch.nan) #[b,t,y]

        #initialize state
        self.t = -1
        self.x = x0
        self.a = self.compute_manifold_latent(self.x, noise=False)
        self.y = self.compute_observation(self.a, noise=False)
        self._log_state()

        return x0, u_seq, num_seqs, num_steps


    def __call__(self, x0=None, u_seq=None, num_seqs=None, num_steps=None):
        """
        inputs: (zeroes by default)
            x0: [b,x], initial latent state, get a0 and y0 from the model
            u_seq: [b,t,u]
        returns:
            x_seq: [b,t,x]
            a_seq: [b,t,a]
            y_seq: [b,t,y]
        """
        x0, u_seq, num_seqs, num_steps = self.init_state(x0, u_seq, num_seqs, num_steps)
        for t in range(1, num_steps):
            self.x_seq[:,t,:] = self.next_dynamic_latent(self.x_seq[:,t-1,:], u_seq[:,t-1,:])
        self.a_seq = self.compute_manifold_latent(self.x_seq)
        self.y_seq = self.compute_observation(self.a_seq)
        return self.x_seq, self.a_seq, self.y_seq


########################
# Nonlinear embeddings #
########################
class NonlinearEmbeddingBase():
    """y = T * e(a)"""
    nonlin_dim = None #Intrinsinc dimension of manifold. Must be set in subclass

    def __init__(self, output_dim=None, lo=-5, hi=5):
        self.output_dim = self.nonlin_dim if output_dim is None else output_dim
        assert self.output_dim >= self.nonlin_dim

        proj_eye = torch.eye(self.nonlin_dim) #first 3 dimensions are identical to embedding, for vizualization
        proj_rand = (hi-lo) * torch.rand(self.output_dim-self.nonlin_dim, self.nonlin_dim) + lo #the other dims are random
        self.projection_matrix = torch.vstack([proj_eye, proj_rand]) #high-dimensional projection matrix


    def __call__(self, a):
        """input: [b,t,a] or [t,a] or [a]
        returns: [b,t,y] or [t,y] or [y]"""
        return (self.projection_matrix @ self.nonlin_embed(a).unsqueeze(-1)).squeeze(-1)


    def inv(self, y):
        """input: [b,t,y] or [t,y] or [y]
        returns: [b,t,n] or [t,n] or [n]"""
        e = torch.pinverse(self.projection_matrix) @ y.mT
        a = self.nonlin_embed_inv(e)
        return a


    def nonlin_embed(self, *args):
        raise NotImplementedError('Override this method')


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

    def nonlin_embed(self, a):
        """input: [b,t,1] or [t,1] or [1]
        returns: [b,t,3] or [t,3] or [3]"""
        e = torch.stack([torch.cos(a),
                         torch.sin(2*a),
                         torch.sin(a)], dim=-1)
        return e


    def __repr__(self):
        return (f'T={self.projection_matrix.numpy()}\n'
                f'e=[cos(a); sin(2a); sin(a)]')


    def plot_manifold(self, ax=None):
        a = torch.linspace(0, 2*torch.pi, 100).unsqueeze(-1)
        return self._plot_manifold(a, ax=ax)



class SwissRoll(NonlinearEmbeddingBase):
    nonlin_dim = 3

    def nonlin_embed(self, a):
        """input: [b,t,2] or [t,2] or [2]
        returns: [b,t,3] or [t,3] or [3]"""
        r, h = a[..., 0], a[..., 1] #radius, height
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

        a = torch.stack([r,h], dim=-1)
        return self._plot_manifold(a, ax=ax)



class Torus(NonlinearEmbeddingBase):
    pass #TODO



class IdentityManifold(NonlinearEmbeddingBase):
    """ Useful for debugging or reducing to a fully linear model"""
    def nonlin_embed(self, a):
        return a

    def nonlin_embed_inv(self, e):
        return e
