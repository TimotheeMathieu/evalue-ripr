import numpy as np
from scipy import stats
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint, root_scalar
from evalue_ripr.constants import EPSILON, DEBUG
from evalue_ripr.proba_tools import fq, lfq, ll, posterior, get_beta_prior, update_posterior, RIP_FW_multi, pwval
from scipy.special import beta as beta_fun
import itertools

class Evalue_GRO():
    def __init__(self, constraints, inits,  real_q,  d):
        self.d = d
        self.n_candidates = n_candidates
        self.evalue = 1
        self.q = real_q
        self.p, kinf = RIP_FW_multi(constraints, inits, 
                                    {"prior_support":[self.q], "prior_weights":[1]},
                                    use_posterior=False)
        
    def observe(self, x):
        x = np.array(x)
        p0val = pwval(self.p, np.array([x])).item()
        self.evalue = self.evalue * (fq(self.q, np.array([x]))/p0val).item()

    def reset(self, seed=None):
        self.evalue = 1

    def reject(self, alpha):
        if self.evalue > 1/alpha:
            return True
        else:
            return None

class POE():
    def __init__(self, constraints, inits, d, M = 50000, prior_data = [], 
                 seed = None):
        self.d = d
        self.n_candidates = n_candidates
        self.cons = contraints
        self.init = inits

        self.M = M
        self.reset(seed = seed, prior_data = prior_data)
        if DEBUG:
            print("Finished computing prior")

    def reset(self, seed = None,prior_factor=None,prior_data=[]):
        self.n_obs = 1
        self.evalue = 1
        self.p = self.init
        self.rng=np.random.RandomState(seed)
        self.q_posterior = 1/2*np.ones(self.d)
        self.posterior = get_beta_prior(self.cons, self.init,self.d, self.rng, M=self.M)
        if len(prior_data)>0:
            for x in prior_data:
                self.update_posterior(x)
                self.q_posterior = (self.n_obs * self.q_posterior + x) / (self.n_obs + 1)
                self.n_obs += 1
    def observe(self, x):
        x = np.array(x)
        self.p, kinf = RIP_FW_multi(self.cons, self.init, self.posterior)
        p0val = pwval(self.p, np.array([x])).item()

        self.evalue = self.evalue * (self.compute_posterior(x)/p0val)
        if DEBUG:
            print("[INFO] Step of POE:",
                  f"x:{x}",
                  f"posterior:{self.compute_posterior(np.array(x)):4e}", 
                  f"fq(p):{fq(self.p, np.array([x])).item():4e}",
                  f"fq(qpost):{fq(self.q_posterior, np.array([x])).item():4e}",
                  f"p:{self.p}, qpost:{self.q_posterior}"
              )

        self.update_posterior(x)
        self.q_posterior = (self.n_obs * self.q_posterior + x) / (self.n_obs + 1)
        self.n_obs += 1

    def update_posterior(self, x):
        self.posterior["alpha"] += x
        self.posterior["beta"] += 1-x
        self.posterior = update_posterior(self.posterior, self.M, self.d, self.cons, self.rng)

    def compute_posterior(self, x):
        return posterior(self.posterior, x,self.cons, self.M,  self.rng)

    def reject(self, alpha):
        if self.evalue > 1/alpha:
            return True
        else:
            return None
