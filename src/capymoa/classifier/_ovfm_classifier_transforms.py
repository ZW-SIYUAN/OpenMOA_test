"""OVFM Transform Functions - Gaussian Copula transformations for mixed data"""
import numpy as np
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF


class OnlineTransformFunction:
    """Transform function for VFS (capricious) streams with fixed dimension"""
    
    def __init__(self, cont_indices, ord_indices, X=None, window_size=100):
        self.cont_indices = cont_indices
        self.ord_indices = ord_indices
        p = len(cont_indices)
        self.window_size = window_size
        self.window = np.array([[np.nan for x in range(p)] for y in range(self.window_size)]).astype(np.float64)
        self.update_pos = np.zeros(p).astype(np.int32)
        if X is not None:
            self.partial_fit(X)
    
    def partial_fit(self, X_batch):
        """Update sliding window with new batch"""
        # Initialize window on first batch
        if np.isnan(self.window[0, 0]):
            # Continuous columns
            mean_cont = np.nanmean(X_batch[:, self.cont_indices])
            std_cont = np.nanstd(X_batch[:, self.cont_indices])
            if np.isnan(mean_cont):
                self.window[:, self.cont_indices] = np.random.normal(
                    0, 1, size=(self.window_size, np.sum(self.cont_indices))
                )
            else:
                self.window[:, self.cont_indices] = np.random.normal(
                    mean_cont, std_cont, size=(self.window_size, np.sum(self.cont_indices))
                )
            
            # Ordinal columns
            for j, loc in enumerate(self.ord_indices):
                if loc:
                    min_ord = np.nanmin(X_batch[:, j])
                    max_ord = np.nanmax(X_batch[:, j])
                    if np.isnan(min_ord):
                        self.window[:, j].fill(0)
                    else:
                        self.window[:, j] = np.random.randint(
                            min_ord, max_ord + 1, size=self.window_size
                        )
        
        # Update window with batch data
        for row in X_batch:
            for col_num in range(len(row)):
                data = row[col_num]
                if not np.isnan(data):
                    self.window[self.update_pos[col_num], col_num] = data
                    self.update_pos[col_num] += 1
                    if self.update_pos[col_num] >= self.window_size:
                        self.update_pos[col_num] = 0

    def partial_evaluate_cont_latent(self, X_batch):
        """Transform continuous features to latent normal space"""
        X_cont = X_batch[:, self.cont_indices]
        window_cont = self.window[:, self.cont_indices]
        Z_cont = np.empty(X_cont.shape)
        Z_cont[:] = np.nan
        
        for i in range(np.sum(self.cont_indices)):
            missing = np.isnan(X_cont[:, i])
            Z_cont[~missing, i] = self.get_cont_latent(X_cont[~missing, i], window_cont[:, i])
        return Z_cont

    def partial_evaluate_ord_latent(self, X_batch):
        """Transform ordinal features to latent intervals"""
        X_ord = X_batch[:, self.ord_indices]
        window_ord = self.window[:, self.ord_indices]
        Z_ord_lower = np.empty(X_ord.shape)
        Z_ord_lower[:] = np.nan
        Z_ord_upper = np.empty(X_ord.shape)
        Z_ord_upper[:] = np.nan
        
        for i in range(np.sum(self.ord_indices)):
            missing = np.isnan(X_ord[:, i])
            Z_ord_lower[~missing, i], Z_ord_upper[~missing, i] = self.get_ord_latent(
                X_ord[~missing, i], window_ord[:, i]
            )
        return Z_ord_lower, Z_ord_upper

    def partial_evaluate_cont_observed(self, Z_batch, X_batch=None):
        """Transform latent to observed continuous features"""
        Z_cont = Z_batch[:, self.cont_indices]
        if X_batch is None:
            X_batch = np.zeros(Z_batch.shape) * np.nan
        X_cont = X_batch[:, self.cont_indices]
        X_cont_imp = np.copy(X_cont)
        window_cont = self.window[:, self.cont_indices]
        
        for i in range(np.sum(self.cont_indices)):
            missing = np.isnan(X_cont[:, i])
            if np.sum(missing) > 0:
                X_cont_imp[missing, i] = self.get_cont_observed(
                    Z_cont[missing, i], window_cont[:, i]
                )
        return X_cont_imp

    def partial_evaluate_ord_observed(self, Z_batch, X_batch=None):
        """Transform latent to observed ordinal features"""
        Z_ord = Z_batch[:, self.ord_indices]
        if X_batch is None:
            X_batch = np.zeros(Z_batch.shape) * np.nan
        X_ord = X_batch[:, self.ord_indices]
        X_ord_imp = np.copy(X_ord)
        window_ord = self.window[:, self.ord_indices]
        
        for i in range(np.sum(self.ord_indices)):
            missing = np.isnan(X_ord[:, i])
            if np.sum(missing) > 0:
                X_ord_imp[missing, i] = self.get_ord_observed(
                    Z_ord[missing, i], window_ord[:, i]
                )
        return X_ord_imp

    def get_cont_latent(self, x_batch_obs, window):
        """Map continuous values to standard normal via empirical CDF"""
        ecdf = ECDF(window)
        l = len(window)
        q = (l / (l + 1.0)) * ecdf(x_batch_obs)
        q[q == 0] = l / (l + 1) / 2
        return norm.ppf(q)

    def get_cont_observed(self, z_batch_missing, window):
        """Map standard normal to continuous values via quantiles"""
        quantiles = norm.cdf(z_batch_missing)
        return np.quantile(window, quantiles)

    def get_ord_latent(self, x_batch_obs, window):
        """Map ordinal values to latent intervals"""
        ecdf = ECDF(window)
        unique = np.unique(window)
        if unique.shape[0] > 1:
            threshold = np.min(np.abs(unique[1:] - unique[:-1])) / 2.0
            z_lower_obs = norm.ppf(ecdf(x_batch_obs - threshold))
            z_upper_obs = norm.ppf(ecdf(x_batch_obs + threshold))
        else:
            z_upper_obs = np.inf
            z_lower_obs = -np.inf
        return z_lower_obs, z_upper_obs

    def get_ord_observed(self, z_batch_missing, window, DECIMAL_PRECISION=3):
        """Map latent values to ordinal categories"""
        n = len(window)
        x = norm.cdf(z_batch_missing)
        quantile_indices = np.ceil(np.round((n + 1) * x - 1, DECIMAL_PRECISION))
        quantile_indices = np.clip(quantile_indices, a_min=0, a_max=n - 1).astype(int)
        sort = np.sort(window)
        return sort[quantile_indices]


class TrapezoidalTransformFunction2:
    """Transform function for TDS (trapezoidal) streams with growing dimension"""
    
    def __init__(self, cont_indices, ord_indices, X=None, window_size=100, window_width=1):
        self.cont_indices = cont_indices
        self.ord_indices = ord_indices
        self.window_width = window_width
        self.window_size = window_size
        self.window = np.array([[np.nan for x in range(window_width)] for y in range(self.window_size)]).astype(np.float64)
        self.update_pos = np.zeros(window_width).astype(np.int32)
        if X is not None:
            self.partial_fit(X)
    
    def partial_fit(self, X_batch, cont_indices, ord_indices):
        """Update window with dynamic dimension expansion"""
        self.cont_indices = cont_indices
        self.ord_indices = ord_indices
        
        # Initialize window on first batch
        if np.isnan(self.window[0, 0]):
            mean_cont = np.nanmean(X_batch[:, self.cont_indices])
            std_cont = np.nanstd(X_batch[:, self.cont_indices])
            
            if np.isnan(mean_cont):
                self.window[:, self.cont_indices] = np.random.normal(
                    0, 1, size=(self.window_size, np.sum(self.cont_indices))
                )
            else:
                self.window[:, self.cont_indices] = np.random.normal(
                    mean_cont, std_cont, size=(self.window_size, np.sum(self.cont_indices))
                )
            
            # Ordinal columns
            for j, loc in enumerate(self.ord_indices):
                if loc:
                    min_ord = np.nanmin(X_batch[:, j])
                    max_ord = np.nanmax(X_batch[:, j])
                    if np.isnan(min_ord):
                        self.window[:, j].fill(0)
                    else:
                        self.window[:, j] = np.random.randint(
                            min_ord, max_ord + 1, size=self.window_size
                        )
            
            # Update with first batch
            for row in X_batch:
                for col_num in range(len(row)):
                    data = row[col_num]
                    if not np.isnan(data):
                        self.window[self.update_pos[col_num], col_num] = data
                        self.update_pos[col_num] += 1
                        if self.update_pos[col_num] >= self.window_size:
                            self.update_pos[col_num] = 0
        else:
            # Expand window if new features appear
            if self.window_width < len(X_batch[-1]):
                add_column = len(X_batch[-1]) - self.window_width
                temp = np.array([[np.nan for x in range(add_column)] for y in range(self.window_size)]).astype(np.float64)
                
                for i in range(add_column):
                    if self.cont_indices[i + self.window_width]:
                        mean_cont = np.nanmean(X_batch[:, self.window_width + i])
                        std_cont = np.nanstd(X_batch[:, self.window_width + i])
                        temp[:, i] = np.random.normal(mean_cont, std_cont, size=(self.window_size))
                    else:
                        min_ord = np.nanmin(X_batch[:, self.window_width + i])
                        max_ord = np.nanmax(X_batch[:, self.window_width + i])
                        temp[:, i] = np.random.randint(min_ord, max_ord + 1, size=self.window_size)
                
                self.window = np.concatenate((self.window, temp), axis=1)
                self.window_width = len(X_batch[-1])
                temp_pos = np.zeros(add_column).astype(np.int32)
                self.update_pos = np.concatenate((self.update_pos, temp_pos), axis=0)
            
            # Update with latest row
            row = X_batch[-1]
            for col_num in range(len(row)):
                data = row[col_num]
                if not np.isnan(data):
                    self.window[self.update_pos[col_num], col_num] = data
                    self.update_pos[col_num] += 1
                    if self.update_pos[col_num] >= self.window_size:
                        self.update_pos[col_num] = 0

    def partial_evaluate_cont_latent(self, X_batch):
        """Transform continuous features to latent space"""
        X_cont = X_batch[:, self.cont_indices]
        window_cont = self.window[:, self.cont_indices]
        Z_cont = np.empty(X_cont.shape)
        Z_cont[:] = np.nan
        
        for i in range(np.sum(self.cont_indices)):
            missing = np.isnan(X_cont[:, i])
            Z_cont[~missing, i] = self.get_cont_latent(X_cont[~missing, i], window_cont[:, i])
        return Z_cont

    def partial_evaluate_ord_latent(self, X_batch):
        """Transform ordinal features to latent intervals"""
        X_ord = X_batch[:, self.ord_indices]
        window_ord = self.window[:, self.ord_indices]
        Z_ord_lower = np.empty(X_ord.shape)
        Z_ord_lower[:] = np.nan
        Z_ord_upper = np.empty(X_ord.shape)
        Z_ord_upper[:] = np.nan
        
        for i in range(np.sum(self.ord_indices)):
            missing = np.isnan(X_ord[:, i])
            Z_ord_lower[~missing, i], Z_ord_upper[~missing, i] = self.get_ord_latent(
                X_ord[~missing, i], window_ord[:, i]
            )
        return Z_ord_lower, Z_ord_upper

    def partial_evaluate_cont_observed(self, Z_batch, X_batch=None):
        """Transform latent to observed continuous"""
        Z_cont = Z_batch[:, self.cont_indices]
        if X_batch is None:
            X_batch = np.zeros(Z_batch.shape) * np.nan
        X_cont = X_batch[:, self.cont_indices]
        X_cont_imp = np.copy(X_cont)
        window_cont = self.window[:, self.cont_indices]
        
        for i in range(np.sum(self.cont_indices)):
            missing = np.isnan(X_cont[:, i])
            if np.sum(missing) > 0:
                X_cont_imp[missing, i] = self.get_cont_observed(Z_cont[missing, i], window_cont[:, i])
        return X_cont_imp

    def partial_evaluate_ord_observed(self, Z_batch, X_batch=None):
        """Transform latent to observed ordinal"""
        Z_ord = Z_batch[:, self.ord_indices]
        if X_batch is None:
            X_batch = np.zeros(Z_batch.shape) * np.nan
        X_ord = X_batch[:, self.ord_indices]
        X_ord_imp = np.copy(X_ord)
        window_ord = self.window[:, self.ord_indices]
        
        for i in range(np.sum(self.ord_indices)):
            missing = np.isnan(X_ord[:, i])
            if np.sum(missing) > 0:
                X_ord_imp[missing, i] = self.get_ord_observed(Z_ord[missing, i], window_ord[:, i])
        return X_ord_imp

    def get_cont_latent(self, x_batch_obs, window):
        """Map continuous to latent normal"""
        ecdf = ECDF(window)
        l = len(window)
        q = (l / (l + 1.0)) * ecdf(x_batch_obs)
        q[q == 0] = l / (l + 1) / 2
        return norm.ppf(q)

    def get_cont_observed(self, z_batch_missing, window):
        """Map latent normal to continuous"""
        quantiles = norm.cdf(z_batch_missing)
        return np.quantile(window, quantiles)

    def get_ord_latent(self, x_batch_obs, window):
        """Map ordinal to latent intervals"""
        ecdf = ECDF(window)
        unique = np.unique(window)
        if unique.shape[0] > 1:
            threshold = np.min(np.abs(unique[1:] - unique[:-1])) / 2.0
            z_lower_obs = norm.ppf(ecdf(x_batch_obs - threshold))
            z_upper_obs = norm.ppf(ecdf(x_batch_obs + threshold))
        else:
            z_upper_obs = np.inf
            z_lower_obs = -np.inf
        return z_lower_obs, z_upper_obs

    def get_ord_observed(self, z_batch_missing, window, DECIMAL_PRECISION=3):
        """Map latent to ordinal categories"""
        n = len(window)
        x = norm.cdf(z_batch_missing)
        quantile_indices = np.ceil(np.round((n + 1) * x - 1, DECIMAL_PRECISION))
        quantile_indices = np.clip(quantile_indices, a_min=0, a_max=n - 1).astype(int)
        sort = np.sort(window)
        return sort[quantile_indices]