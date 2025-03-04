class DFINE(nn.Module):

    def forward(self, y, u, mask=None):
        # Get the dimensions from y
        num_seq, num_steps, _ = y.shape

        # Create the mask if it's None
        if mask is None:
            mask = torch.ones(y.shape[:-1]).unsqueeze(dim=-1)

        # Get the encoded low-dimensional manifold factors (project via nonlinear manifold embedding)
        a_hat = self.encoder(y)

        # Run LDM to infer filtered and smoothed dynamic latent factors
        x_pred, x_filter, x_smooth, Lambda_pred, Lambda_filter, Lambda_smooth = self.ldm(a=a_hat, u=u, mask=mask, do_smoothing=True)
        a_pred = (self.ldm.C @ x_pred.unsqueeze(dim=-1)).squeeze(dim=-1) #  (num_seq, num_steps, dim_a, dim_x) x (num_seq, num_steps, dim_x, 1) --> (num_seq, num_steps, dim_a)
        a_filter = (self.ldm.C @ x_filter.unsqueeze(dim=-1)).squeeze(dim=-1) #  (num_seq, num_steps, dim_a, dim_x) x (num_seq, num_steps, dim_x, 1) --> (num_seq, num_steps, dim_a)
        a_smooth = (self.ldm.C @ x_smooth.unsqueeze(dim=-1)).squeeze(dim=-1) #  (num_seq, num_steps, dim_a, dim_x) x (num_seq, num_steps, dim_x, 1) --> (num_seq, num_steps, dim_a)

        # Remove the last timestep of predictions since it's T+1|T, which is not of interest to us
        x_pred = x_pred[:, :-1, :]
        Lambda_pred = Lambda_pred[:, :-1, :, :]
        a_pred = a_pred[:, :-1, :]

        if self.config.model.fit_D_matrix:
            a_pred = a_pred + (self.ldm.D @ u[:,1:,...].unsqueeze(dim=-1)).squeeze(dim=-1) #first index of a_pred is a_{1|0}, therefore, it has to be summed with u_{1}
            a_filter = a_filter + (self.ldm.D @ u.unsqueeze(dim=-1)).squeeze(dim=-1)
            a_smooth = a_smooth + (self.ldm.D @ u.unsqueeze(dim=-1)).squeeze(dim=-1)

        # Supervise a_seq or a_smooth to behavior if requested -> behv_hat shape: (num_seq, num_steps, dim_behv)
        if self.config.model.supervise_behv:
            if self.config.model.behv_from_smooth:
                behv_hat = self.mapper(a_smooth.reshape(-1, self.dim_a))
            else:
                behv_hat = self.mapper(a_hat.reshape(-1, self.dim_a))
            behv_hat = behv_hat.reshape(-1, num_steps, self.dim_behv)
        else:
            behv_hat = None

        # Get filtered and smoothed estimates of neural observations
        y_hat = self.decoder(a_hat.reshape(-1, self.dim_a)).reshape(num_seq, -1, self.dim_y)
        y_pred = self.decoder(a_pred.reshape(-1, self.dim_a)).reshape(num_seq, -1, self.dim_y)
        y_filter = self.decoder(a_filter.reshape(-1, self.dim_a)).reshape(num_seq, -1, self.dim_y)
        y_smooth = self.decoder(a_smooth.reshape(-1, self.dim_a)).reshape(num_seq, -1, self.dim_y)

        # Dump inferrred latents, predictions and reconstructions to a dictionary
        model_vars = dict(a_hat=a_hat, a_pred=a_pred, a_filter=a_filter, a_smooth=a_smooth,
                          x_pred=x_pred, x_filter=x_filter, x_smooth=x_smooth,
                          Lambda_pred=Lambda_pred, Lambda_filter=Lambda_filter, Lambda_smooth=Lambda_smooth,
                          y_hat=y_hat, y_pred=y_pred, y_filter=y_filter, y_smooth=y_smooth,
                          behv_hat=behv_hat)
        return model_vars


    def get_k_step_ahead_prediction(self, model_vars, k, u):
        '''
        Performs k-step ahead prediction of manifold latent factors, dynamic latent factors and neural observations.
        '''
        # Check whether provided k value is valid or not
        assert k>0 and isinstance(k, int), 'Number of steps ahead prediction value is invalid or of wrong type, k must be a positive integer!'

        # Extract the required variables from model_vars dictionary
        x_filter = model_vars['x_filter'] #[b,t,x]

        # Get the required dimensions
        num_seq, num_steps, _ = x_filter.shape

        # Here is where k-step ahead prediction is iteratively performed
        x_pred_k = x_filter[:, :-k, ...] #[x_{0|0}, x_{1|1}, ..., x_{(T-k)|(T-k)}], will contain [x_{k|0}, x_{(k+1)|1}, ..., x_{T|(T-k)}]
        for i in range(0, k):
            s = slice(i, num_steps-k+i)
            x_pred_k = (self.ldm.A @ x_pred_k.unsqueeze(dim=-1) + self.ldm.B @ u[:,s].unsqueeze(dim=-1)).squeeze(dim=-1)


        a_pred_k = (self.ldm.C @ x_pred_k.unsqueeze(dim=-1)).squeeze(dim=-1)
        if self.config.model.fit_D_matrix:
            a_pred_k = a_pred_k + (self.ldm.D @ u[:,k:,...].unsqueeze(dim=-1)).squeeze(dim=-1)

        # After obtaining k-step ahead predicted manifold latent factors, they're decoded to obtain k-step ahead predicted neural observations
        decoder = model_vars['decoder'] if 'decoder' in model_vars else self.decoder
        y_pred_k = decoder(a_pred_k)

        #sanity check
        if k == 1:
            assert (y_pred_k == model_vars['y_pred']).all()
            assert (x_pred_k == model_vars['x_pred']).all()
            assert (a_pred_k == model_vars['a_pred']).all()

        return y_pred_k, a_pred_k, x_pred_k


    def compute_loss(self, y, u, model_vars, mask=None, behv=None, y_hat_cl=None, y_target_cl=None):

        # Create the mask if it's None
        if mask is None:
            mask = torch.ones(y.shape[:-1]).unsqueeze(dim=-1)

        # Dump individual loss values for logging or Tensorboard
        loss_dict = dict()

        # Iterate over multiple steps ahead
        k_steps_mse_sum = torch.tensor(0.)
        y_pred_all = {}
        for k in self.config.loss.steps_ahead:
            #TODO this can be made k times faster by reusing previous k's computation
            y_pred_k, _, _ = self.get_k_step_ahead_prediction(model_vars, k=k, u=u.detach())
            #TODO: if I don't detach u, how will this affect this k-step-ahead computation graph?

            y_pred_all[k] = y_pred_k
            mse_pred = compute_mse(y_flat=y[:, k:, :].reshape(-1, self.dim_y),
                                    y_hat_flat=y_pred_k.reshape(-1, self.dim_y),
                                    mask_flat=mask[:, k:, :].reshape(-1,))
            k_steps_mse_sum += mse_pred
            loss_dict[f'steps_{k}_mse'] = mse_pred

        model_loss = k_steps_mse_sum
        loss_dict['model_loss'] = model_loss


        # Get MSE loss for behavior reconstruction, 0 if we dont supervise our model with behavior data
        behv_mse = torch.tensor(0.)
        behv_loss = torch.tensor(0.)
        if self.config.model.supervise_behv:
            behv_mse = compute_mse(y_flat=behv[..., self.config.model.which_behv_dims].reshape(-1, self.dim_behv),
                                   y_hat_flat=model_vars['behv_hat'].reshape(-1, self.dim_behv),
                                   mask_flat=mask.reshape(-1,))
            behv_loss = self.scale_behv_recons * behv_mse
        loss_dict['behv_mse'] = behv_mse
        loss_dict['behv_loss'] = behv_loss


        # L2 regularization loss
        reg_loss = torch.tensor(0.)
        if self.scale_l2 > 0:
            for name, param in self.named_parameters():
                if 'weight' in name:
                    reg_loss = reg_loss + self.scale_l2 * torch.norm(param)
            loss_dict['reg_loss'] = reg_loss


        # B matrix inverse spectral norm regularization loss
        spectr_reg_B_loss = torch.tensor(0.)
        if self.scale_spectr_reg_B > 0:
            # spectr_reg_B_loss = self.scale_spectr_reg_B / torch.linalg.svd(self.ldm.B)[1].min()
            spectr_reg_B_loss = self.scale_spectr_reg_B * (torch.linalg.cond(self.ldm.B) - 1)
            loss_dict['spectr_reg_B_loss'] = spectr_reg_B_loss


        # Control loss
        control_loss = torch.tensor(0.)
        if y_target_cl is not None and self.scale_control_loss > 0:
            y_hat_T = y_hat_cl[:,-1,:] #model_vars['y_filter'][:,-1,:]  #[b,y]

            control_loss = F.mse_loss(y_hat_T, y_target_cl)

            control_loss = self.scale_control_loss * control_loss
            loss_dict['control_loss'] = control_loss

        # Final loss
        loss = model_loss + behv_loss + reg_loss + spectr_reg_B_loss + control_loss
        loss_dict['total_loss'] = loss
        return loss, loss_dict, y_pred_all


class LDM(nn.Module):
    '''
    Linear Dynamical Model backbone for DFINE. This module is used for smoothing and filtering
    given a batch of trials/segments/time-series.

    LDM equations are as follows:
    x_{t+1} = A*x_t + B*u_t + w_t , cov(w_t) = W
    a_t     = C*x_t + r_t         , cov(r_t) = R
    '''

    def compute_forwards(self, a, u, mask=None):
        '''
        Performs the forward iteration of causal flexible Kalman filtering, given a batch of trials/segments/time-series
        '''
        num_seq, num_steps, _ = a.shape

        if mask is None:
            mask = torch.ones(num_seq, num_steps)

        # Make sure that mask is 3D (last axis is 1-dimensional)
        if len(mask.shape) != len(a.shape):
            mask = mask.unsqueeze(dim=-1) # (num_seq, num_steps, 1)

        # To make sure we do not accidentally use the real outputs in the steps with missing values, set them to a dummy value, e.g., 0.
        # The dummy values of observations at masked points are irrelevant because:
        # Kalman disregards the observations by setting Kalman Gain to 0 in K = torch.mul(K, mask[:, t, ...].unsqueeze(dim=1)) @ line 205
        a_masked = torch.mul(a, mask) # (num_seq, num_steps, dim_a) x (num_seq, num_steps, 1)

        # Initialize mu_0 and Lambda_0
        mu_pred = self.mu_0.unsqueeze(dim=0).repeat(num_seq, 1) # (num_seq, dim_x)
        Lambda_pred = self.Lambda_0.unsqueeze(dim=0).repeat(num_seq, 1, 1) # (num_seq, dim_x, dim_x)

        # Filtered and predicted estimates/error covariance. NOTE: The last time-step of the prediction has T+1|T, which may not be of interest
        mu_t_all = [] #mu_t_all[t,i,:] = mu_{t|t} for t=0..T-1
        Lambda_t_all = []
        mu_pred_all = [] #mu_pred_all[t,i,:] = mu_{t+1|t} for t=0..T-1
        Lambda_pred_all = []

        # Get covariance matrices
        W, R = self._get_covariance_matrices()

        for t in range(num_steps):
            # Obtain residual
            a_pred = (self.C @ mu_pred.unsqueeze(dim=-1)).squeeze(dim=-1) # (num_seq, dim_a)
            r = a_masked[:, t, ...] - a_pred # (num_seq, dim_a)
            if self.fit_D_matrix:
                r = r - (self.D @ u[:,t,...].unsqueeze(dim=-1)).squeeze(dim=-1) # (num_seq, dim_a)

            # Project system uncertainty into measurement space, get Kalman Gain
            S = self.C @ Lambda_pred @ self.C.T + R # (num_seq, dim_a, dim_a)
            S_inv = torch.inverse(S) # (num_seq, dim_a, dim_a)
            K = Lambda_pred @ self.C.T @ S_inv # (num_seq, dim_x, dim_a)
            K = torch.mul(K, mask[:, t, ...].unsqueeze(dim=1))  # (num_seq, dim_x, dim_a) x (num_seq, 1,  1)

            # Get current mu and Lambda
            mu_t = mu_pred + (K @ r.unsqueeze(dim=-1)).squeeze(dim=-1) # (num_seq, dim_x)
            I_KC = torch.eye(self.dim_x) - K @ self.C # (num_seq, dim_x, dim_x)
            Lambda_t = I_KC @ Lambda_pred # (num_seq, dim_x, dim_x)

            # Prediction
            if t < num_steps-1:
                u_t = u[:, t, ...]
            else:
                #last timestep of predictions will be dropped since it's T+1|T so give it dummy input
                u_t = torch.full((num_seq, self.dim_u), torch.nan)

            mu_pred = (self.A @ mu_t.unsqueeze(dim=-1) + self.B @ u_t.unsqueeze(dim=-1)).squeeze(dim=-1) #(num_seq, dim_x, dim_x) x (num_seq, dim_x, 1) + (num_seq, dim_x, dim_u) x (num_seq, dim_u, 1) --> (num_seq, dim_x, 1) --> (num_seq, dim_x)
            Lambda_pred = self.A @ Lambda_t @ self.A.T + W #(num_seq, dim_x, dim_x) x (num_seq, dim_x, dim_x) x (num_seq, dim_x, dim_x) --> (num_seq, dim_x, dim_x)

            # Store predictions and updates
            mu_pred_all.append(mu_pred)
            mu_t_all.append(mu_t)
            Lambda_pred_all.append(Lambda_pred)
            Lambda_t_all.append(Lambda_t)

        mu_pred_all = torch.stack(mu_pred_all, dim=0) #[t,b,x]
        mu_t_all = torch.stack(mu_t_all, dim=0) #[t,b,x]
        Lambda_pred_all = torch.stack(Lambda_pred_all, dim=0) #[t,b,x,x]
        Lambda_t_all = torch.stack(Lambda_t_all, dim=0) #[t,b,x,x]

        return mu_pred_all, mu_t_all, Lambda_pred_all, Lambda_t_all


    def filter(self, a, u=None, mask=None):
        '''
        Performs Kalman Filtering
        '''

        # Run the forward iteration
        mu_pred_all, mu_t_all, Lambda_pred_all, Lambda_t_all = self.compute_forwards(a=a, u=u, mask=mask)

        # Swap num_seq and num_steps dimensions
        mu_pred_all = torch.permute(mu_pred_all, (1, 0, 2))
        mu_t_all = torch.permute(mu_t_all, (1, 0, 2))
        Lambda_pred_all = torch.permute(Lambda_pred_all, (1, 0, 2, 3))
        Lambda_t_all = torch.permute(Lambda_t_all, (1, 0, 2, 3))

        return mu_pred_all, mu_t_all, Lambda_pred_all, Lambda_t_all



    def forward(self, a, u=None, mask=None):
        mu_pred_all, mu_t_all, Lambda_pred_all, Lambda_t_all = self.filter(a=a, u=u, mask=mask)
        mu_back_all = torch.ones_like(mu_t_all, device=mu_t_all.device)
        Lambda_back_all = torch.ones_like(Lambda_t_all, device=Lambda_t_all.device)

        return mu_pred_all, mu_t_all, mu_back_all, Lambda_pred_all, Lambda_t_all, Lambda_back_all




target_generator = TargetGenerator(num_targets=model_config['config']['train.batch_size'],
                                   rnn=rnn, rnn_train_data=rnn_train_data)
controller = make_controller(trainer.dfine, **control_config)
closed_loop = make_closed_loop(plant=rnn, dfine=trainer.dfine, controller=controller, copy_dfine=False)
train_data = ControlledDFINEDataset(closed_loop, target_generator, num_steps=run_config['num_steps'])


# Train the DFINE model
torch.autograd.set_detect_anomaly(True)
train_loader = DataLoader(train_data, batch_size=model_config['config']['train.batch_size'])
trainer.train(train_loader)


class LinearQuadratic(Controller):
    """Base class for LQR"""
    def __init__(self, A, B, C, b=0, Q=1, R=1, Qf=None, u_min=-float('inf'), u_max=float('inf'), include_u_ss=True, penalize_obs=True):
        self.A = A
        self.B = B
        self.C = C
        self.b = b
        self.dim_x, self.dim_u = B.shape

        self.Q, self.R, self.Qf = self.make_cost_matrices(Q, R, Qf, penalize_obs)

        self.u_min = u_min
        self.u_max = u_max
        self.include_u_ss = include_u_ss


    def init_target(self, x_target):
        """
        x_target: [b,x]
        """
        super().init_target(x_target)
        self.u_target = self.compute_steady_state_control(x_target)
        return self.u_target


    def compute_steady_state_control(self, x_ss):
        if not self.include_u_ss:
            num_seqs, dim_x = verify_shape(x_ss, [None, self.dim_x])
            return torch.zeros(num_seqs, self.dim_u)

        I = torch.eye(self.dim_x) #[x,x]
        IAx = ((I-self.A) @ x_ss.unsqueeze(-1) - self.b)

        pinvB = torch.linalg.pinv(self.B)
        u_ss = (pinvB @ IAx).squeeze(-1) #[u,x] @ ([x,x]@[b,x,1] - [b,x,1]) -> [b,u,1] -> [b,u]

        return u_ss



class LinearQuadraticRegulator(LinearQuadratic):
    def __init__(self, *args, horizon=float('inf'), **kwargs):
        r"""
        System:
            x_{t+1} = A x_t + B u_t
            y_t = C x_t

        Control objective:
            if T < inf:
                J = x_T Qf x_T + \sum_{t=1}^{T-1} x_t Q x_t + u_t R u_t
            else:
                J = \sum_t x_t Q x_t + u_t R u_t

        u_min, u_max: ad-hoc hard bounds on the control input
        """
        super().__init__( *args, **kwargs)
        self.K = self.compute_lqr_gain(horizon)


    def compute_lqr_gain(self, horizon=float('inf')):
        """Compute LQR feedback gain matrix K"""
        P = [None] * (horizon + 1)
        P[horizon] = self.Qf
        K = [None] * horizon
        for k in range(horizon-1, -1, -1):
            K[k] = torch.linalg.solve(self.R + self.B.T @ P[k+1] @ self.B, self.B.T @ P[k+1] @ self.A)
            P[k] = self.Q + self.A.T @ P[k+1] @ self.A - self.A.T @ P[k+1] @ self.B @ K[k]
        return K

    def __call__(self, x):
        K = self.K[self.t]
        u = (-K @ (x - self.x_target).unsqueeze(-1)).squeeze(-1) + self.u_target #[u,x] @ [b,x,1] + [b,u] -> [b,u]
        self.t += 1
        return u
