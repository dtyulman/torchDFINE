from copy import deepcopy

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import control

from python_utils import verify_shape, verify_output_dim, linspace
from time_series_utils import generate_input_noise, compute_control_error
from plot_utils import plot_parametric, plot_vs_time, plot_heatmap
from DFINE import DFINE


class NonlinearStateSpaceModel():
    def __init__(self, **kwargs):
        """
        Dynamic latent:  x_{t+1} = A x_t + b + B u_t + q_t,  q_t ~ N(0,Q)
        Manifold latent:     a_t = C x_t + D u_t + r_t,      r_t ~ N(0,R)
        Observation:         y_t = f(a_t) + s_t,             s_t ~ N(0,S)

        kwargs:
            A, b, B, C, D, f: Model parameters. Revert to defaults if not specified.
            dim_x, dim_a, dim_y, dim_u: Must specify relevant dimensions if relying on defaults
            Q, R, S: Noise covariance matrices. If scalars, scales identity by value. Set to None for no noise.
        """
        #TODO: these should be properties so that they dont need to be updated if any of the matrices change
        self.dim_x = kwargs.pop('dim_x', None)
        self.dim_a = kwargs.pop('dim_a', None)
        self.dim_y = kwargs.pop('dim_y', None)
        self.dim_u = kwargs.pop('dim_u', None)

        # get params if they have been specified
        self.A = kwargs.pop('A', None)
        self.b = kwargs.pop('b', None)
        self.B = kwargs.pop('B', None)
        self.C = kwargs.pop('C', None)
        self.D = kwargs.pop('D', None)
        self.f = kwargs.pop('f', None)

        # if specified param, infer dims from it, or verify that it matches specified dim
        self.dim_x, _          = verify_shape(self.A, [self.dim_x, self.dim_x])
        self.dim_x,            = verify_shape(self.b, [self.dim_x])
        self.dim_x, self.dim_u = verify_shape(self.B, [self.dim_x, self.dim_u])
        self.dim_a, self.dim_x = verify_shape(self.C, [self.dim_a, self.dim_x])
        self.dim_a, self.dim_u = verify_shape(self.D, [self.dim_a, self.dim_u])
        self.dim_y             = verify_output_dim(self.f, self.dim_a, self.dim_y)

        # set defaults for unspecified params using specified/inferred dims
        self.A = torch.eye(self.dim_x, self.dim_x)     if self.A is None else self.A  #[x,x]
        self.b = torch.zeros(self.dim_x)               if self.b is None else self.b  #[x]
        self.B = torch.eye(self.dim_x, self.dim_u)     if self.B is None else self.B  #[x,u]
        self.C = torch.eye(self.dim_a, self.dim_x)     if self.C is None else self.C  #[a,x]
        self.D = torch.zeros(self.dim_a, self.dim_u)   if self.D is None else self.D  #[a,u]
        self.f = LinearManifold(torch.eye(self.dim_y)) if self.f is None else self.f  #(a->y)

        # noise distributions
        self.Q_distr = make_noise_distr(kwargs.pop('Q', 1e-2), self.dim_x) #[x,x]
        self.R_distr = make_noise_distr(kwargs.pop('R', 1e-2), self.dim_a) #[a,a]
        self.S_distr = make_noise_distr(kwargs.pop('S', 2e-3), self.dim_y) #[y,y]

        # logging and state, initialized in init_state()
        self.x_seq = self.a_seq = self.y_seq = None
        self.x = self.a = self.y = self.t = None


    def __repr__(self):
        return (
            f'A={self.A.numpy()}\n'
            f'b={self.b.numpy()}\n'
            f'B={self.B.numpy()}\n'
            f'Q={self.Q_distr.covariance_matrix.numpy() if self.Q_distr is not None else 0}\n'

            '--\n'

            f'C={self.C.numpy()}\n'
            f'R={self.R_distr.covariance_matrix.numpy() if self.R_distr is not None else 0}\n'

            '--\n'

            f'{self.f}\n'
            f'S={self.S_distr.covariance_matrix.numpy() if self.S_distr is not None else 0}\n'
            )


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


    def compute_next_dynamic_latent(self, x, u=None, noise=True):
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


    def get_observation(self):
        return self.y


    def _update_state(self, u, noise=True):
        self.x = self.compute_next_dynamic_latent(self.x, u, noise=noise)
        self.a = self.compute_manifold_latent(self.x, noise=noise)
        self.y = self.compute_observation(self.a, noise=noise)
        return self.x, self.a, self.y


    def _log_state(self):
        self.t += 1 #increment this first, ensures that x_seq[:,self.t,:] = self.x
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
        self._log_state() #increments self.t

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
    input_dim = None
    nonlin_dim = None #intrinsinc dimension of manifold, must be set in subclass
    default_input_lims = None #default range of inputs for manifold, e.g. for plotting, setting control target, etc., must be set in subclass

    def __init__(self, output_dim=None, lo=-5, hi=5):
        self.output_dim = output_dim or self.nonlin_dim
        assert self.output_dim >= self.nonlin_dim
        assert len(self.default_input_lims) == self.input_dim

        proj_eye = torch.eye(self.nonlin_dim) #first 3 dimensions are identical to embedding, for vizualization
        proj_rand = (hi-lo) * torch.rand(self.output_dim-self.nonlin_dim, self.nonlin_dim) + lo #the other dims are random
        self.projection_matrix = torch.vstack([proj_eye, proj_rand]) #high-dimensional projection matrix


    def __repr__(self):
        return (f'{self.__class__.__name__}\n'
                f'T={self.projection_matrix.numpy()}\n')


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
    input_dim = 1
    nonlin_dim = 3
    default_input_lims = [(0, 2*torch.pi)]

    def nonlin_embed(self, a):
        """input: [b,t,1] or [t,1] or [1]
        returns: [b,t,3] or [t,3] or [3]"""
        a = a.squeeze(-1)
        e = torch.stack([torch.cos(a),
                         torch.sin(2*a),
                         torch.sin(a)], dim=-1)
        return e


    def plot_manifold(self, samples=100, ax=None):
        a = torch.linspace(0, 2*torch.pi, samples).unsqueeze(-1)
        return self._plot_manifold(a, ax=ax)



class SwissRoll(NonlinearEmbeddingBase):
    input_dim = 2
    nonlin_dim = 3
    default_input_lims = [(0, 3*torch.pi), ((-5, 5))]

    def nonlin_embed(self, a):
        """input: [b,t,2] or [t,2] or [2]
        returns: [b,t,3] or [t,3] or [3]"""
        r, h = a[..., 0], a[..., 1] #radius, height
        e = torch.stack([0.5*r*torch.cos(r),
                         h,
                         0.5*r*torch.sin(r)], dim=-1)
        return e


    # def nonlin_embed_inv(self, e):
    #     h = e[..., 1]
    #     r =
    #     return torch.stack([r,
    #                         h])

    def plot_manifold(self, rlim=None, hlim=None, r_samples=300, h_samples=30, ax=None):
        rlim = rlim or self.default_input_lims[0]
        hlim = hlim or self.default_input_lims[1]

        r = torch.linspace(rlim[0], rlim[1], r_samples)
        h = torch.linspace(hlim[0], hlim[1], h_samples)

        r,h = torch.meshgrid(r, h) #TODO: see torch.cartesian_prod() instead
        r,h = r.flatten().unsqueeze(-1), h.flatten().unsqueeze(-1)

        a = torch.stack([r,h], dim=-1)
        return self._plot_manifold(a, ax=ax)



class Torus(NonlinearEmbeddingBase):
    input_dim = 2
    nonlin_dim = 3
    default_input_lims = [(0, 2*torch.pi), ((0, 2*torch.pi))]

    def __init__(self, r1=4, r2=1.5, *args, **kwargs):
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


    def plot_manifold(self, r1_samples=50, r2_samples=125, ax=None, mode='scatter'):
        a1 = linspace(0, 2*torch.pi, r1_samples, endpoint=mode=='surf')
        a2 = linspace(0, 2*torch.pi, r2_samples, endpoint=mode=='surf')

        a1,a2 = torch.meshgrid(a1, a2) #TODO: see torch.cartesian_prod() instead
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


class LinearManifold(NonlinearEmbeddingBase):
    def __init__(self, M):
        self.M = M
        self.output_dim, self.input_dim = M.shape
        self.nonlin_dim = self.output_dim

        self.default_input_lims = [(-10, 10)]*self.input_dim

        super().__init__(self.output_dim)


    def nonlin_embed(self, a):
        return (self.M @ a.unsqueeze(-1)).squeeze(-1)


    def nonlin_embed_inv(self, e):
        return (torch.pinv(self.M) @ e.unsqueeze(-1)).squeeze(-1)


    def plot_manifold(self, a_lims=None, a_samples=50, ax=None):
        """
        a_lims:
            tuple (lo, hi): makes same limits for each input dimension
            list of tuples [(lo1, hi1), (lo2, hi2)]: lims for each input dim
        a_samples:
            int: same number of samples for each dimension
            list of ints [s1, s2]: number of samples for each input dim
        """
        if 2 > self.output_dim > 3 or self.input_dim > 2:
            raise RuntimeError("Can't plot for larger than 2D input and 3D output")

        if a_lims is None:
            a_lims = self.default_input_lims
        a_lims = [a_lims]*self.input_dim if isinstance(a_lims, tuple) else a_lims

        a_samples = [a_samples]*self.input_dim if isinstance(a_samples, int) else a_samples

        a = [linspace(a_lims[i][0], a_lims[i][1], a_samples[i]) for i in range(self.input_dim)]
        a = torch.stack(torch.meshgrid(*a), dim=-1) #TODO: see torch.cartesian_prod() instead

        return self._plot_manifold(a, ax=ax)






###########
# Helpers #
###########

def make_noise_distr(M, dim):
    if M is not None:
        if isinstance(M, (float, int)):
            M = M * torch.eye(dim) #[x,x]
        verify_shape(M, [dim, dim])
        M = MultivariateNormal(torch.zeros(dim), M)
    return M


def make_ssm(spec, **kwargs):
    # https://realpython.com/factory-method-python/
    if spec == 'ring':
        assert kwargs.pop('dim_a', 1) == 1
        return NonlinearStateSpaceModel(dim_x = kwargs.pop('dim_x', 1),
                                        dim_a = 1,
                                        dim_y = kwargs.pop('dim_y', 3),
                                        dim_u = kwargs.pop('dim_u', 1),
                                        f = RingManifold(),
                                        **kwargs)

    elif spec == 'swiss':
        assert kwargs.pop('dim_a', 2) == 2
        return NonlinearStateSpaceModel(dim_x = kwargs.pop('dim_x', 2),
                                        dim_a = 2,
                                        dim_y = kwargs.pop('dim_y', 3),
                                        dim_u = kwargs.pop('dim_u', 2),
                                        f = SwissRoll(),
                                        **kwargs)

    elif spec == 'torus':
        assert kwargs.pop('dim_a', 2) == 2
        return NonlinearStateSpaceModel(dim_x = kwargs.pop('dim_x', 2),
                                        dim_a = 2,
                                        dim_y = kwargs.pop('dim_y', 3),
                                        dim_u = kwargs.pop('dim_u', 2),
                                        f = Torus(),
                                        **kwargs)

    elif spec == 'linear':
        dim_a = kwargs.pop('dim_a', None)
        dim_y = kwargs.pop('dim_y', None)
        M = kwargs.pop('M', torch.randn(dim_y, dim_a))
        dim_y, dim_a = verify_shape(M, [dim_y, dim_a])
        return NonlinearStateSpaceModel(dim_x = kwargs.pop('dim_x', 2),
                                        dim_a = dim_a,
                                        dim_y = dim_y,
                                        dim_u = kwargs.pop('dim_u', 2),
                                        f = LinearManifold(M),
                                        **kwargs)

    elif spec == 'identity':
        dim_a = kwargs.pop('dim_a', 2)
        assert kwargs.pop('dim_y', dim_y) == dim_a, "dim_y must be equal to dim_a for identity manifold"
        return NonlinearStateSpaceModel(dim_x = kwargs.pop('dim_x', 2),
                                        dim_a = dim_a,
                                        dim_y = dim_a,
                                        dim_u = kwargs.pop('dim_u', 2),
                                        f = LinearManifold(torch.eye(dim_a)),
                                        **kwargs)

    elif isinstance(spec, DFINE):
        assert len(kwargs) == 0, 'Do not specify additional kwargs' #TODO: allow overwriting via kwargs?
        Q,R = spec.ldm._get_covariance_matrices()
        params = dict(A = spec.ldm.A,
                      B = spec.ldm.B,
                      C = spec.ldm.C,
                      f = spec.decoder,
                      Q = Q,
                      R = R,
                      S = None)
        return NonlinearStateSpaceModel(**params)

    else:
        raise ValueError(spec)


def make_dfine_from_ssm(ssm, dfine):
    # class WrapperModule(torch.nn.Module):
    #     def __init__(self, f):
    #         super().__init__()
    #         self.f = f

    #     def forward(self, *args, **kwargs):
    #         return self.f(*args, **kwargs)

    assert dfine.ldm.dim_x == ssm.dim_x
    assert dfine.ldm.dim_a == ssm.dim_a
    assert dfine.ldm.dim_y == ssm.dim_y
    assert dfine.ldm.dim_u == ssm.dim_u

    dfine = deepcopy(dfine)
    with torch.no_grad:
        dfine.ldm.A.data = ssm.A
        dfine.ldm.B.data = ssm.B
        dfine.ldm.C.data = ssm.C
        dfine.ldm.W_log_diag.data = torch.diag(torch.log(ssm.Q_distr.covariance_matrix))
        dfine.ldm.R_log_diag.data = torch.diag(torch.log(ssm.R_distr.covariance_matrix))
        #TODO: don't remember whether/why this was necessary, remove if not. Maybe to register it for saving the model
        # dfine.decoder = WrapperModule(ssm.f)
        # dfine.encoder = WrapperModule(ssm.f.inv)
        dfine.decoder = ssm.f
        dfine.encoder = ssm.f.inv
    return dfine


def generate_dataset(ssm, num_seqs, num_steps, x0_min=None, x0_max=None, **noise_params):
    u_train = generate_input_noise(ssm.dim_u, num_seqs, num_steps, **noise_params) #[b,t,u]

    #TODO: technically ssm.f.default_input_lims are lims on `a` not `x`
    x0_min = x0_min or torch.tensor([lims[0] for lims in ssm.f.default_input_lims])
    x0_max = x0_max or torch.tensor([lims[1] for lims in ssm.f.default_input_lims])

    x0 = (x0_max - x0_min) * torch.rand(num_seqs, ssm.dim_x) + x0_min #[b,x]
    x_train, a_train, y_train = ssm(x0=x0, u_seq=u_train)
    return x_train, a_train, y_train, u_train


def plot_data_sample(x=None, a=None, y=None, f=None, num_seqs=10, cbar=True):
    varlist = [(s,v) for s,v in [('x',x), ('a',a), ('y',y)] if v is not None]
    n_axs = len(varlist)
    fig = plt.figure()
    for ax_idx, (varname, var) in enumerate(varlist):
        if var is not None:
            B,T,N = var.shape
            ax = fig.add_subplot(1, n_axs, ax_idx+1, projection='3d' if N>=3 else None)
            if varname == 'y' and N>1 and f is not None:
                if isinstance(f, SwissRoll):
                    f.plot_manifold(ax=ax,
                                    rlim=[a[:num_seqs,:,0].min(), a[:num_seqs,:,0].max()],
                                    hlim=[a[:num_seqs,:,1].min(), a[:num_seqs,:,1].max()])
                else:
                    f.plot_manifold(ax=ax)

            for i in range(num_seqs):
                if N == 1:
                    plot_vs_time(var[i], varname=varname, legend=False, ax=ax)
                else:
                    plot_parametric(var[i], mode='line', ax=ax, varname=varname,
                                    cbar=cbar and ax_idx==0 and i==num_seqs-1)
                    ax.axis('equal')
    fig.set_size_inches(n_axs*4,4)
    fig.tight_layout()
    return fig, fig.get_axes()


def make_target(ssm, x=None, endpoint=True):
    """
    x:
        tensor [b,x]: Single target or multiple targets (i.e. b=1 or >1). Ensures dtype is float.
        list or tuple [x]: Single target. Converts this to a [1,x] shaped tensor of floats.
        list of tuples [(x1_lo, x1_hi, x1_samples), ..., (xN_lo, xN_hi, xN_samples)]:
            range of x_samples targets linearly spaced between x_lo and x_hi in each dimension
    """
    if x is None:
        x = [(lims[0], lims[1], 40) for lims in ssm.f.default_input_lims]

    if isinstance(x, list) and isinstance(x[0], tuple) and len(x[0])==3:
        x = torch.cartesian_prod(*[linspace(lo,hi,n, endpoint=endpoint) for lo,hi,n in x]) #[b,x]
    elif isinstance(x, (list, tuple)) and isinstance(x[0], (int, float)):
        x = torch.tensor(x, dtype=torch.get_default_dtype()).unsqueeze(0) #[1,x]

    a = ssm.compute_manifold_latent(x)
    y = ssm.compute_observation(a)
    return x, a, y


def plot_error_heatmap(output, target, x_target, t=-1, varname=''):
    """
    output: [b,t,v]
    target: [b,v]
    x_target: [b,t,v]. Used to compute ax ticks
    """
    error = compute_control_error(output, target, t=t)

    h_axis = x_target[:,0].unique()
    v_axis = x_target[:,1].unique()
    error = error.reshape(len(h_axis), len(v_axis))

    fig, ax = plot_heatmap(error, h_axis, v_axis)
    ax.set_title(f'Error ${varname}$')
    ax.set_xlabel('Target $x^{*}_0$')
    ax.set_ylabel('Target $x^{*}_1$')
    fig.tight_layout()

    return fig, ax, error
