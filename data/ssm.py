import numpy as np
import matplotlib.pyplot as plt


class NonlinearStateSpaceModel():
    """
    z_{t+1} = A*z_t + B*u_t + q_t,   q_t ~ N(0,Q)
    a_t     = C*z_t + r_t,           w_t ~ N(0,R) 
    y_t     = f(a_t) + s_t,          r_t ~ N(0,S)
    """
    def __init__(self, A,B,C,f, Q,R,S):        
        self.A = A #[z,z]
        self.B = B #[z,u]
        self.C = C #[a,z]
        self.f = f #[y,a]
        
        self.Q = Q #[z,z]
        self.R = R #[a,a]
        self.S = S #[y,y]
                
        #sanity check dimensions
        assert A.shape[0] == A.shape[1] #z
        #TODO
        #...
        
        #convenience variables
        self.dim_a, self.dim_z = C.shape        
        self.dim_y = len(self.f(np.zeros(self.dim_a)))
        
                  
    def _step_z(self, z, u=None):
        q = np.random.multivariate_normal( np.zeros(self.dim_z), self.Q )
        z_next = self.A @ z + q
        if u:
            z_next += self.B @ u 
        return z_next
    
    
    def _compute_a(self, z):
        r = np.random.multivariate_normal( np.zeros(self.dim_a), self.R )
        a = self.C @ z + r
        return a
    
    
    def _compute_y(self, a):
        s = np.random.multivariate_normal( np.zeros(self.dim_y), self.S )
        y = self.f( a ) + s
        return y
    
    
    def step(self, z, u=None):     
        z_next = self._step_z(z, u)
        a_next = self._compute_a(z_next)
        y_next = self._compute_y(a_next)      
        return z_next, a_next, y_next
    
    
    def generate_trajectory(self, z0, u_seq=None, num_steps=None, num_seqs=None):
        if num_steps is None:
            num_steps = len(u_seq)

        _num_seqs = 1 if num_seqs is None else num_seqs
                        
        z_seq = np.empty((_num_seqs, num_steps, self.dim_z)) #[b,t,z]
        a_seq = np.empty((_num_seqs, num_steps, self.dim_a)) #[b,t,a]
        y_seq = np.empty((_num_seqs, num_steps, self.dim_y)) #[b,t,y]
        if num_seqs is None:
            z_seq, a_seq, y_seq = z_seq.squeeze(0), a_seq.squeeze(0), y_seq.squeeze(0)   
        
        z_seq[0] = z0       
        a_seq[0] = self._compute_a(z_seq[0])
        y_seq[0] = self._compute_y(a_seq[0])
        
        for t in range(1, num_steps):
            z_seq[t], a_seq[t], y_seq[t] = self.step(z_seq[t-1], u_seq[t-1])     


         
        return z_seq, a_seq, y_seq 
    
    
     
###########           
# Helpers #
###########  
def plot_y(y_seq, ax=None, plot_mode='line'):
    #plots only first 3 dimensions

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    if plot_mode == 'line':
        for i in range(len(y_seq)-1):
            cmap_idx = int(255*i/len(y_seq))
            ax.plot(y_seq[i:i+2, 0], y_seq[i:i+2, 1], y_seq[i:i+2, 2], color=plt.cm.jet(cmap_idx))
    elif plot_mode == 'scatter':    
        im = ax.scatter(y_seq[:,0],y_seq[:,1],y_seq[:,2], c=range(len(y_seq)), cmap='jet')
        fig.colorbar(im, ax=ax, label='Time')
        
    ax.set_xlabel('$y_1$')
    ax.set_ylabel('$y_2$')        
    ax.set_zlabel('$y_3$')
    
    
def plot_z(z_seq, ax=None):
    fig, ax = plt.subplots()
    ax.plot(z_seq)
    ax.set_xlabel('Time')
    ax.set_ylabel('z')


def make_nonlinear_embedding_fn(output_dim, manifold_type='ring'): #TODO: opportunity for factory pattern!      
    #define nonlinear embedding     
    if manifold_type == 'ring':
        input_dim = 1
        nonlin_embed = lambda d: np.array([np.cos(d), np.sin(2*d), np.sin(d)])
    elif manifold_type == 'swiss_roll':
        input_dim = 2
        nonlin_embed = lambda dr, dh: np.array([0.5*dr*np.cos(dr), dh, 0.5*dr*np.sin(dr)])
    else:
        raise ValueError('Invalid manifold type')
    
    #project nonlinear embedding into high dimensions to get final observation
    nonlin_dim = len(nonlin_embed(np.ones((input_dim,1))))       
    assert output_dim >= nonlin_dim
 
    T_eye = np.eye(nonlin_dim) #first 3 dimensions are identical to embedding, for vizualization
    lo, hi = -5, 5       
    T_rand = (hi-lo) * np.random.rand(output_dim-nonlin_dim, nonlin_dim) + lo
    T = np.vstack([T_eye, T_rand]) #high-dimensional projection matrix
    
    f = lambda a: T @ nonlin_embed(*a)
    return f
            

#########
# Usage #
#########
if __name__ == '__main__':
    dim_z = 1
    dim_u = 1
    dim_a = 1
    dim_y = 3
    
    A = np.array([[0.99]])
    B = np.array([[1]])
    C = np.eye(dim_z)
    f = make_nonlinear_embedding_fn(dim_y, manifold_type='ring')
    
    Q = 0 * np.diag(np.random.rand(dim_z))
    R = np.zeros((dim_a, dim_a))
    S = 0.01 * np.diag(np.random.rand(dim_y))
    
    ssm = NonlinearStateSpaceModel(A,B,C,f, Q,R,S)
    
    num_steps = 100
    u = np.ones((num_steps, dim_u)) * 0.2
    z0 = np.zeros(dim_z)
    z,a,y = ssm.generate_trajectory(z0=z0, u_seq=u)

    plot_z(z)
    plot_y(y)
