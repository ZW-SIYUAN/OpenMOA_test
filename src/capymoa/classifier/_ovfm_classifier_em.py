"""OVFM EM Algorithm Components - Extracted from original OVFM code"""
import numpy as np
from scipy.stats import norm, truncnorm
from concurrent.futures import ProcessPoolExecutor


def _em_step_body_(args):
    """Wrapper for parallel execution"""
    return _em_step_body(*args)


def _em_step_body(Z, r_lower, r_upper, sigma, num_ord_updates=1):
    """EM step for a batch of instances"""
    num, p = Z.shape
    Z_imp = np.copy(Z)
    C = np.zeros((p, p))
    for i in range(num):
        c, z_imp, z = _em_step_body_row(Z[i, :], r_lower[i, :], r_upper[i, :], sigma)
        Z_imp[i, :] = z_imp
        Z[i, :] = z
        C += c
    return C, Z_imp, Z


def _em_step_body_row(Z_row, r_lower_row, r_upper_row, sigma, num_ord_updates=1):
    """EM step for a single instance"""
    Z_imp_row = np.copy(Z_row)
    p = Z_imp_row.shape[0]
    num_ord = r_upper_row.shape[0]
    C = np.zeros((p, p))

    obs_indices = np.where(~np.isnan(Z_row))[0]
    missing_indices = np.setdiff1d(np.arange(p), obs_indices)
    ord_in_obs = np.where(obs_indices < num_ord)[0]
    ord_obs_indices = obs_indices[ord_in_obs]

    sigma_obs_obs = sigma[np.ix_(obs_indices, obs_indices)]
    sigma_obs_missing = sigma[np.ix_(obs_indices, missing_indices)]
    sigma_missing_missing = sigma[np.ix_(missing_indices, missing_indices)]

    if len(missing_indices) > 0:
        tot_matrix = np.concatenate((np.identity(len(sigma_obs_obs)), sigma_obs_missing), axis=1)
        intermed_matrix = np.linalg.solve(sigma_obs_obs, tot_matrix)
        sigma_obs_obs_inv = intermed_matrix[:, :len(sigma_obs_obs)]
        J_obs_missing = intermed_matrix[:, len(sigma_obs_obs):]
    else:
        sigma_obs_obs_inv = np.linalg.solve(sigma_obs_obs, np.identity(len(sigma_obs_obs)))

    var_ordinal = np.zeros(p)

    # Update ordinal latent variables
    if len(obs_indices) >= 2 and len(ord_obs_indices) >= 1:
        for update_iter in range(num_ord_updates):
            sigma_obs_obs_inv_Z_row = np.dot(sigma_obs_obs_inv, Z_row[obs_indices])
            for ind in range(len(ord_obs_indices)):
                j = obs_indices[ind]
                v = sigma_obs_obs_inv[:, ind]
                new_var_ij = 1.0 / v[ind]
                new_mean_ij = Z_row[j] - new_var_ij * sigma_obs_obs_inv_Z_row[ind]
                
                mean, var = truncnorm.stats(
                    a=(r_lower_row[j] - new_mean_ij) / np.sqrt(new_var_ij),
                    b=(r_upper_row[j] - new_mean_ij) / np.sqrt(new_var_ij),
                    loc=new_mean_ij,
                    scale=np.sqrt(new_var_ij),
                    moments='mv'
                )
                
                if np.isfinite(var):
                    var_ordinal[j] = var
                    if update_iter == num_ord_updates - 1:
                        C[j, j] = C[j, j] + var
                if np.isfinite(mean):
                    Z_row[j] = mean

    # Impute missing values
    Z_obs = Z_row[obs_indices]
    Z_imp_row[obs_indices] = Z_obs
    if len(missing_indices) > 0:
        Z_imp_row[missing_indices] = np.matmul(J_obs_missing.T, Z_obs)
        if len(ord_obs_indices) >= 1 and len(obs_indices) >= 2 and np.sum(var_ordinal) > 0:
            cov_missing_obs_ord = J_obs_missing[ord_in_obs].T * var_ordinal[ord_obs_indices]
            C[np.ix_(missing_indices, ord_obs_indices)] += cov_missing_obs_ord
            C[np.ix_(ord_obs_indices, missing_indices)] += cov_missing_obs_ord.T
            C[np.ix_(missing_indices, missing_indices)] += (
                sigma_missing_missing 
                - np.matmul(J_obs_missing.T, sigma_obs_missing)
                + np.matmul(cov_missing_obs_ord, J_obs_missing[ord_in_obs])
            )
        else:
            C[np.ix_(missing_indices, missing_indices)] += (
                sigma_missing_missing - np.matmul(J_obs_missing.T, sigma_obs_missing)
            )
    
    return C, Z_imp_row, Z_row


class ExpectationMaximizationBase:
    """Base class for EM algorithms"""
    
    def _project_to_correlation(self, covariance):
        """Project covariance to correlation matrix"""
        D = np.diagonal(covariance)
        D_neg_half = 1.0 / np.sqrt(D)
        covariance *= D_neg_half
        return covariance.T * D_neg_half

    def _init_Z_ord(self, Z_ord_lower, Z_ord_upper, seed):
        """Initialize ordinal latent variables"""
        Z_ord = np.empty(Z_ord_lower.shape)
        Z_ord[:] = np.nan

        n, k = Z_ord.shape
        obs_indices = ~np.isnan(Z_ord_lower)

        u_lower = np.copy(Z_ord_lower)
        u_lower[obs_indices] = norm.cdf(Z_ord_lower[obs_indices])
        u_upper = np.copy(Z_ord_upper)
        u_upper[obs_indices] = norm.cdf(Z_ord_upper[obs_indices])

        np.random.seed(seed)
        for i in range(n):
            for j in range(k):
                if not np.isnan(Z_ord_upper[i, j]) and u_upper[i, j] > 0 and u_lower[i, j] < 1:
                    u_sample = np.random.uniform(u_lower[i, j], u_upper[i, j])
                    Z_ord[i, j] = norm.ppf(u_sample)
        return Z_ord


class OnlineExpectationMaximization(ExpectationMaximizationBase):
    """Online EM for VFS (capricious) streams"""
    
    def __init__(self, cont_indices, ord_indices, window_size=200, sigma_init=None):
        self.cont_indices = cont_indices
        self.ord_indices = ord_indices
        p = len(cont_indices)
        self.sigma = sigma_init if sigma_init is not None else np.identity(p)
        self.window_size = window_size
        self.iteration = 1

    def partial_fit_and_predict(self, X_batch, max_workers=1, num_ord_updates=2, 
                                decay_coef=0.5, sigma_update=True, marginal_update=True, 
                                sigma_out=False):
        """Update and predict for a batch"""
        from capymoa.classifier._ovfm_classifier_transforms import OnlineTransformFunction
        
        # Lazy initialization of transform function
        if not hasattr(self, 'transform_function'):
            self.transform_function = OnlineTransformFunction(
                self.cont_indices, self.ord_indices, window_size=self.window_size
            )
        
        if marginal_update:
            self.transform_function.partial_fit(X_batch)
        
        res = self._fit_covariance(X_batch, max_workers, num_ord_updates, 
                                   decay_coef, sigma_update, sigma_out)
        
        if sigma_out:
            Z_batch_imp, sigma = res
        else:
            Z_batch_imp = res
        
        # Rearrange to match original feature order
        Z_imp_rearranged = np.empty(X_batch.shape)
        Z_imp_rearranged[:, self.ord_indices] = Z_batch_imp[:, :np.sum(self.ord_indices)]
        Z_imp_rearranged[:, self.cont_indices] = Z_batch_imp[:, np.sum(self.ord_indices):]
        
        X_imp = np.empty(X_batch.shape)
        X_imp[:, self.cont_indices] = self.transform_function.partial_evaluate_cont_observed(
            Z_imp_rearranged, X_batch
        )
        X_imp[:, self.ord_indices] = self.transform_function.partial_evaluate_ord_observed(
            Z_imp_rearranged, X_batch
        )
        
        if sigma_out:
            return Z_imp_rearranged, X_imp, sigma
        return Z_imp_rearranged, X_imp

    def _fit_covariance(self, X_batch, max_workers=1, num_ord_updates=2, 
                       decay_coef=0.5, update=True, sigma_out=False, seed=1):
        """Fit covariance using online EM"""
        Z_ord_lower, Z_ord_upper = self.transform_function.partial_evaluate_ord_latent(X_batch)
        Z_ord = self._init_Z_ord(Z_ord_lower, Z_ord_upper, seed)
        Z_cont = self.transform_function.partial_evaluate_cont_latent(X_batch)
        
        Z = np.concatenate((Z_ord, Z_cont), axis=1)
        batch_size, p = Z.shape
        prev_sigma = self.sigma
        
        Z_imp = np.zeros((batch_size, p))
        C = np.zeros((p, p))
        
        if max_workers == 1:
            C, Z_imp, Z = _em_step_body(Z, Z_ord_lower, Z_ord_upper, prev_sigma, num_ord_updates)
        else:
            divide = (batch_size / max_workers * np.arange(max_workers + 1)).astype(int)
            args = [(Z[divide[i]:divide[i+1], :], 
                    Z_ord_lower[divide[i]:divide[i+1], :],
                    Z_ord_upper[divide[i]:divide[i+1], :], 
                    prev_sigma, num_ord_updates) for i in range(max_workers)]
            
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                res = pool.map(_em_step_body_, args)
                for i, (C_divide, Z_imp_divide, Z_divide) in enumerate(res):
                    Z_imp[divide[i]:divide[i+1], :] = Z_imp_divide
                    Z[divide[i]:divide[i+1], :] = Z_divide
                    C += C_divide
        
        C = C / batch_size
        sigma = np.cov(Z_imp, rowvar=False) + C
        sigma = self._project_to_correlation(sigma)
        
        if update:
            self.sigma = sigma * decay_coef + (1 - decay_coef) * prev_sigma
            self.iteration += 1
        
        if sigma_out:
            sigma_out_val = self.get_sigma(self.sigma if update else 
                                          sigma * decay_coef + (1 - decay_coef) * prev_sigma)
            return Z_imp, sigma_out_val
        return Z_imp

    def get_sigma(self, sigma=None):
        """Get correlation matrix in original feature order"""
        if sigma is None:
            sigma = self.sigma
        sigma_rearranged = np.empty(sigma.shape)
        sigma_rearranged[np.ix_(self.ord_indices, self.ord_indices)] = (
            sigma[:np.sum(self.ord_indices), :np.sum(self.ord_indices)]
        )
        sigma_rearranged[np.ix_(self.cont_indices, self.cont_indices)] = (
            sigma[np.sum(self.ord_indices):, np.sum(self.ord_indices):]
        )
        sigma_rearranged[np.ix_(self.cont_indices, self.ord_indices)] = (
            sigma[np.sum(self.ord_indices):, :np.sum(self.ord_indices)]
        )
        sigma_rearranged[np.ix_(self.ord_indices, self.cont_indices)] = (
            sigma_rearranged[np.ix_(self.cont_indices, self.ord_indices)].T
        )
        return sigma_rearranged


class TrapezoidalExpectationMaximization2(ExpectationMaximizationBase):
    """Online EM for TDS (trapezoidal) streams with dynamic dimension growth"""
    
    def __init__(self, cont_indices, ord_indices, window_size, window_width, sigma_init=None):
        self.cont_indices = cont_indices
        self.ord_indices = ord_indices
        self.window_size = window_size
        p = len(cont_indices)
        self.sigma = sigma_init if sigma_init is not None else np.identity(p)
        self.iteration = 1

    def partial_fit_and_predict(self, X_batch, cont_indices, ord_indices, max_workers=1,
                                num_ord_updates=2, decay_coef=0.5, sigma_update=True,
                                marginal_update=True, sigma_out=False):
        """Update and predict for a batch with growing dimensions"""
        from capymoa.classifier._ovfm_classifier_transforms import TrapezoidalTransformFunction2
        
        self.cont_indices = cont_indices
        self.ord_indices = ord_indices
        
        # Lazy initialization
        if not hasattr(self, 'transform_function'):
            self.transform_function = TrapezoidalTransformFunction2(
                cont_indices, ord_indices, 
                window_size=self.window_size,
                window_width=len(X_batch[0])
            )
        
        if marginal_update:
            self.transform_function.partial_fit(X_batch, cont_indices, ord_indices)
        
        res = self._fit_covariance(X_batch, max_workers, num_ord_updates,
                                   decay_coef, sigma_update, sigma_out)
        
        if sigma_out:
            Z_batch_imp, sigma = res
        else:
            Z_batch_imp = res
        
        Z_imp_rearranged = np.empty(X_batch.shape)
        Z_imp_rearranged[:, self.ord_indices] = Z_batch_imp[:, :np.sum(self.ord_indices)]
        Z_imp_rearranged[:, self.cont_indices] = Z_batch_imp[:, np.sum(self.ord_indices):]
        
        X_imp = np.empty(X_batch.shape)
        X_imp[:, self.cont_indices] = self.transform_function.partial_evaluate_cont_observed(
            Z_imp_rearranged, X_batch
        )
        X_imp[:, self.ord_indices] = self.transform_function.partial_evaluate_ord_observed(
            Z_imp_rearranged, X_batch
        )
        
        if sigma_out:
            return Z_imp_rearranged, X_imp, sigma
        return Z_imp_rearranged, X_imp

    def _fit_covariance(self, X_batch, max_workers=1, num_ord_updates=2,
                       decay_coef=0.5, update=True, sigma_out=False, seed=1):
        """Fit covariance with dimension expansion"""
        Z_ord_lower, Z_ord_upper = self.transform_function.partial_evaluate_ord_latent(X_batch)
        Z_ord = self._init_Z_ord(Z_ord_lower, Z_ord_upper, seed)
        Z_cont = self.transform_function.partial_evaluate_cont_latent(X_batch)
        
        Z = np.concatenate((Z_ord, Z_cont), axis=1)
        batch_size, p = Z.shape
        prev_sigma = self.sigma
        s = prev_sigma.shape[0]
        
        # Expand sigma if needed
        if p > s:
            temp = np.zeros([s, p - s])
            prev_sigma = np.insert(prev_sigma, [s], temp, axis=1)
            temp = np.zeros([p - s, p])
            prev_sigma = np.insert(prev_sigma, [s], temp, axis=0)
            for i in range(s, p):
                prev_sigma[i][i] = 1.0
        
        Z_imp = np.zeros((batch_size, p))
        C = np.zeros((p, p))
        
        if max_workers == 1:
            C, Z_imp, Z = _em_step_body(Z, Z_ord_lower, Z_ord_upper, prev_sigma, num_ord_updates)
        else:
            divide = (batch_size / max_workers * np.arange(max_workers + 1)).astype(int)
            args = [(Z[divide[i]:divide[i+1], :],
                    Z_ord_lower[divide[i]:divide[i+1], :],
                    Z_ord_upper[divide[i]:divide[i+1], :],
                    prev_sigma, num_ord_updates) for i in range(max_workers)]
            
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                res = pool.map(_em_step_body_, args)
                for i, (C_divide, Z_imp_divide, Z_divide) in enumerate(res):
                    Z_imp[divide[i]:divide[i+1], :] = Z_imp_divide
                    Z[divide[i]:divide[i+1], :] = Z_divide
                    C += C_divide
        
        C = C / batch_size
        sigma = np.cov(Z_imp, rowvar=False) + C
        L = sigma.shape[0]
        for i in range(L):
            if sigma[i][i] == 0:
                sigma[i][i] = 1.0
        sigma = self._project_to_correlation(sigma)
        
        if update:
            self.sigma = sigma * decay_coef + (1 - decay_coef) * prev_sigma
            self.iteration += 1
        
        if sigma_out:
            sigma_out_val = self.get_sigma(self.sigma if update else
                                          sigma * decay_coef + (1 - decay_coef) * prev_sigma)
            return Z_imp, sigma_out_val
        return Z_imp

    def get_sigma(self, sigma=None):
        """Get correlation in original feature order"""
        if sigma is None:
            sigma = self.sigma
        sigma_rearranged = np.empty(sigma.shape)
        sigma_rearranged[np.ix_(self.ord_indices, self.ord_indices)] = (
            sigma[:np.sum(self.ord_indices), :np.sum(self.ord_indices)]
        )
        sigma_rearranged[np.ix_(self.cont_indices, self.cont_indices)] = (
            sigma[np.sum(self.ord_indices):, np.sum(self.ord_indices):]
        )
        sigma_rearranged[np.ix_(self.cont_indices, self.ord_indices)] = (
            sigma[np.sum(self.ord_indices):, :np.sum(self.ord_indices)]
        )
        sigma_rearranged[np.ix_(self.ord_indices, self.cont_indices)] = (
            sigma_rearranged[np.ix_(self.cont_indices, self.ord_indices)].T
        )
        return sigma_rearranged