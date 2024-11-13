import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import control

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
        self.D = torch.zeros(self.dim_a, self.dim_u)
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


    def is_observable(self, verbose=False, method='direct'):
        """
        method:
            'direct' - builds controllability matrix and computes its rank
            'hautus' - uses Hautus Lemma https://en.wikipedia.org/wiki/Hautus_lemma
        """
        if method == 'direct':
            obsv = control.obsv(self.A.numpy(), self.C.numpy())
            rank = np.linalg.matrix_rank(obsv)
            result = rank >= self.dim_x
            if verbose:
                print(f'{"" if result else "not "}observable.'.capitalize(),
                      f'rank(O)={rank} {">=" if result else "<"} n={self.dim_x}')
            return result

        elif method == 'hautus':
            eigvals = torch.linalg.eigvals(self.A)
            for lambda_i in eigvals:
                matrix = np.vstack((lambda_i * np.eye(self.dim_x) - self.A.numpy(), self.C.numpy()))
                if np.linalg.matrix_rank(matrix) < self.dim_x:
                    return False
            return True

        else:
            raise ValueError(f'Invalid method: {method}')


    def is_controllable(self, verbose=False, method='direct'):
        """
        method:
            'direct' - builds controllability matrix and computes its rank
            'hautus' - uses Hautus Lemma https://en.wikipedia.org/wiki/Hautus_lemma
        """
        if method == 'direct':
            ctrb = control.ctrb(self.A.numpy(), self.B.numpy())
            rank = np.linalg.matrix_rank(ctrb)
            result = rank >= self.dim_x
            if verbose:
                print(f'{"" if result else "not "}controllable.'.capitalize(),
                      f'rank(CC)={rank} {">=" if result else "<"} n={self.dim_x}')
            return result

        elif method == 'hautus':
            eigvals = torch.linalg.eigvals(self.A)
            for lambda_i in eigvals:
                matrix = np.hstack((lambda_i * np.eye(self.dim_x) - self.A.numpy(), self.B.numpy()))
                if np.linalg.matrix_rank(matrix) < self.dim_x:
                    return False
            return True

        else:
            raise ValueError(f'Invalid method: {method}')


    def is_output_controllable(self, verbose=False):
        """
        https://en.wikipedia.org/wiki/Controllability#Output_controllability
        """
        ctrb = control.ctrb(self.A.numpy(), self.B.numpy())
        output_ctrb = np.concatenate((self.C.numpy() @ ctrb, self.D.numpy()), axis=1)
        rank = np.linalg.matrix_rank(output_ctrb)
        result = rank >= self.dim_a
        if verbose:
            print(f'{"" if result else "not "}output-controllable.'.capitalize(),
                  f'rank(OC)={rank} {">=" if result else "<"} n={self.dim_a}')
        return result



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
        return a


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
        e = (torch.pinverse(self.projection_matrix) @ y.unsqueeze(-1)).squeeze(-1)
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
        ax.scatter(*y.T, color='grey', alpha=0.2, marker='.', s=2)
        return fig, ax

    def plot_manifold(self):
        raise NotImplementedError('Override this method')



class RingManifold(NonlinearEmbeddingBase):
    nonlin_dim = 3

    def nonlin_embed(self, a):
        """input: [b,t,1] or [t,1] or [1]
        returns: [b,t,3] or [t,3] or [3]"""
        a = a.squeeze(-1)
        e = torch.stack([torch.cos(a),
                         torch.sin(2*a),
                         torch.sin(a)], dim=-1)
        return e


    def __repr__(self):
        return (f'T={self.projection_matrix.numpy()}\n'
                f'e=[cos(a); sin(2a); sin(a)]')


    def plot_manifold(self, samples=100, ax=None):
        a = torch.linspace(0, 2*torch.pi, samples).unsqueeze(-1)
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


    def plot_manifold(self, rlim=(0, 6*torch.pi), hlim=(-1,1), r_samples=300, h_samples=30, ax=None):
        r = torch.linspace(rlim[0], rlim[1], r_samples)
        h = torch.linspace(hlim[0], hlim[1], h_samples)

        r,h = torch.meshgrid(r, h)
        r,h = r.flatten().unsqueeze(-1), h.flatten().unsqueeze(-1)

        a = torch.stack([r,h], dim=-1)
        return self._plot_manifold(a, ax=ax)



class Torus(NonlinearEmbeddingBase):
    nonlin_dim = 3

    def __init__(self, r1, r2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r1 = r1
        self.r2 = r2


    def nonlin_embed(self, a):
        """input: [b,t,2] or [t,2] or [2]
        returns: [b,t,3] or [t,3] or [3]"""
        a1, a2 = a[..., 0], a[..., 1] #major, minor
        e = torch.stack([(self.r1 + self.r2*torch.cos(a1))*torch.cos(a2),
                         (self.r1 + self.r2*torch.cos(a1))*torch.sin(a2),
                         self.r2*torch.sin(a1)], dim=-1)
        return e


    def plot_manifold(self, r1_samples=50, r2_samples=125, ax=None, mode='surf'):
        a1 = torch.linspace(0, 2*torch.pi*(r1_samples-1)/r1_samples, r1_samples)
        a2 = torch.linspace(0, 2*torch.pi*(r2_samples-1)/r2_samples, r2_samples)

        a1,a2 = torch.meshgrid(a1, a2)
        a1,a2 = a1.flatten().unsqueeze(-1), a2.flatten().unsqueeze(-1)

        a = torch.stack([a1,a2], dim=-1)
        if mode == 'scatter':
            fig, ax = self._plot_manifold(a, ax=ax)

        elif mode == 'surf':
            y = self.nonlin_embed(a)

            y1 = y[:,0,0].reshape(r1_samples, r2_samples)
            y2 = y[:,0,1].reshape(r1_samples, r2_samples)
            y3 = y[:,0,2].reshape(r1_samples, r2_samples)

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ls = mpl.colors.LightSource()
            ax.plot_surface(y1,y2,y3, color='gray', shade=True, lightsource=ls, alpha=0.3)

        ax.set_aspect('equal')
        return fig, ax


# def plot_torus(precision, c, a):
#     U = np.linspace(0, 2*np.pi, precision)
#     V = np.linspace(0, 2*np.pi, precision)
#     U, V = np.meshgrid(U, V)
#     X = (c+a*np.cos(V))*np.cos(U)
#     Y = (c+a*np.cos(V))*np.sin(U)
#     Z = a*np.sin(V)
#     return X, Y, Z

class IdentityManifold(NonlinearEmbeddingBase):
    """ Useful for debugging or reducing to a fully linear model"""
    def __init__(self, output_dim):
        self.nonlin_dim = output_dim
        super().__init__(output_dim)

    def nonlin_embed(self, a):
        return a

    def nonlin_embed_inv(self, e):
        return e
