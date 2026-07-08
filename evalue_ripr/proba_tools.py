import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy import stats
from scipy.special import beta as beta_fun
from copy import deepcopy
import itertools
from evalue_ripr.constants import EPSILON, DEBUG

def fq(p, x):
    p = np.clip(p, EPSILON, 1 - EPSILON)
    logprobas = x @ np.log(p) + (1 - x) @ np.log(1 - p)
    return np.exp(logprobas)

def lfq(p, x):
    return np.log(fq(p, x))

def dfq_dp(p, x):
    p = np.clip(p, EPSILON, 1 - EPSILON)
    f = fq(p, x)  # shape: (n_samples,)
    term1 = x / p[np.newaxis, :]  # shape: (n_samples, n_p)
    term2 = (1 - x) / (1 - p[np.newaxis, :])
    dL_dp = term1 - term2  # shape: (n_samples, n_p)
    return f[:, np.newaxis] * dL_dp

def ll(p, mu, N):
    # log likelihood
    if not np.all((p>0) & (p < 1)):
        return -np.inf
    else:
        return np.sum(N*mu* np.log(p+EPSILON) + N*(1-mu) * np.log(1-p+EPSILON))

def kl1(q1, p1):
    """
    Bernoulli KL
    """
    return q1 * np.log(q1/p1) + (1-q1)*np.log((1-q1)/(1-p1))

def kl(q,p):
    return np.sum(kl1(q, p))

def RIP_point(constraints, init, q):
    d = len(q)
    ret = minimize(lambda p : kl(q,p), 
                   x0 = init, 
                   constraints=constraints, 
                   bounds = [(EPSILON,1-EPSILON)]*d)
    if ret.success:
        return ret.x, ret.fun
    else:
        print("Warning, Rip point did not converge")
        return ret.x, ret.fun
    

def get_beta_prior(cons, init, d, rng, M = 1000, alpha = 1/2, beta = 1/2):
    """
    Get independent beta prior conditioned on cons
    """
    prior = {"alpha": alpha*np.ones(d),
             "beta": beta*np.ones(d)}
    prior = update_posterior(prior, M, d, cons, rng)
    return prior

def posterior(pw1, x, cons, M=1000, rng=None):
    """
    x must be in {0,1}^d
    """
    alpha = pw1["alpha"]
    beta = pw1["beta"]
    d = len(alpha)
    result = 0
    samples2 = (x[np.newaxis, :] * pw1["samples"]  + (1-x[np.newaxis, :]) * (1-pw1["samples"]))
    integrand = np.prod(samples2*2, axis=1) # the times 2 is for numerical purpose. divide by 2 at the end
    mean_int = np.mean(integrand)
    norm_cons=0
    is_in_cell = pw1["isin_cell"]

    expectation = (np.sum(integrand[is_in_cell]) + mean_int) / (len(is_in_cell)+1)
    norm_cons = (np.sum(is_in_cell) + 1)/(len(samples2)+1)
    return expectation / norm_cons / 2**d

def update_posterior(prior, M, d , constraints,  rng):
    posterior = deepcopy(prior)
    norm_cons = 0
    samples = np.array([stats.beta(posterior["alpha"][i],
                                   posterior["beta"][i]).rvs(size=[M],random_state=rng) 
                        for i in range(d)]).T

    posterior["samples"] = samples
    distancesu = [ np.max((cons.A @ (samples.T) - cons.ub[:,np.newaxis]).T,   axis=1) for cons in constraints]
    distances = np.min(np.array(distancesu), axis = 0)
    posterior["isin_cell"] = distances>0
    return posterior

def pwval(pw, x):
    res = np.zeros(len(x))
    for i in range(len(pw["prior_weights"])):
        res += fq(pw["prior_support"][i],x) * pw["prior_weights"][i]
    return res
    
def Jq0_multi(p, pw1val, pw0val, xs):
    d = len(p)
    fpval = fq(p, xs)
    res = np.sum(fpval * pw1val / pw0val)
    return res

def DJq0_multi(p, pw1val, pw0val, xs):
    d = len(p)
    dfpval = dfq_dp(p, xs)
    res = np.sum(dfpval * (pw1val / pw0val)[:,np.newaxis])
    return res


def kinf_multi(pw1val, pw0val):
    return np.sum(pw1val[pw1val != 0] * np.log(pw1val[pw1val != 0]  / pw0val[pw1val != 0] ))

def RIP_FW_multi(constraints, inits, pw1, maxiter=300, 
                 tol = 1e-5, use_posterior = True):
    d = len(inits[0])
    xs = np.array(list(itertools.product([0,1], repeat=d)))

    if use_posterior:
        pw1val = np.array([posterior(pw1, x, constraints) for x in xs])
    else:
        pw1val = pwval(pw1, xs)
        
    for j in range(len(inits)):
        # Better initialization using point point projection
        if not use_posterior:
            p = np.zeros(d)
            for k in range(len(pw1["prior_support"])):
                p = p + pw1["prior_support"][k] * pw1["prior_weights"][k]
        else:
            p = pw1["alpha"]/(pw1["alpha"]+pw1["beta"])

        q,_ = RIP_point([constraints[j]], inits[j], p)
        inits[j] = q
        print(q)

    def lmo(pw0):
        minres = np.inf
        qres = inits[0]
        pw0val = pwval(pw0, xs)
        for k in range(len(constraints)):
            ret = minimize(lambda p: -Jq0_multi(p, pw1val, pw0val, xs), 
                           x0 = inits[k], 
                           jac = lambda p: -DJq0_multi(p, pw1val, pw0val, xs),
                           constraints=constraints[k], 
                           bounds = [(EPSILON, 1- EPSILON)]*d,
                           )
            # print(ret)
            if ret.success :
                if ret.fun < minres:
                    qres = ret.x
                    minres = ret.fun
            else:
                ret = minimize(lambda p: -Jq0_multi(p, pw1val, pw0val, xs), 
                           x0 = inits[k], 
                           constraints=constraints[k], 
                           method="cobyla",
                           bounds = [(EPSILON, 1- EPSILON)]*d,
                           )
                if ret.fun < minres:
                    qres = ret.x
                    minres = ret.fun

        if minres == np.inf:
            print("Warning: may have a problem with optimisation")
            return inits[0]
        return qres

    pw0 = {"prior_support":inits,
           "prior_weights":np.ones(len(inits))/len(inits)}

    # print(pw0)
    pw0 = FrankWolfePrior_multi(pw0,
                                lmo,
                                maxiter=maxiter, 
                                tol=tol,
                                pw1val = pw1val)

    return pw0, kinf_multi(pw1val, pwval(pw0, xs))
    
def FrankWolfePrior_multi(init,
                    lmo,
                    maxiter=200, 
                    tol=1e-5,
                    pw1val = None,
                    use_linesearch=True
                    ):
    pw = init # initial value of prior
    qold = np.average(pw["prior_support"], weights=pw["prior_weights"], axis=0) # for stopping criterion

    support = pw["prior_support"]
    weights = pw["prior_weights"]

    for t in range(maxiter):
        # Linear optimization step
        qopt = lmo(pw)
        # FW step size
        alpha = 2/(2+t)

        # Line search. Most of the time this is useful.
        if use_linesearch:
            support2 = np.vstack([support, [qopt]])
            d = len(pw["prior_support"][0])
            xs = np.array(list(itertools.product([0,1], repeat=d)))

            def obj(alpha):
                w2 = np.hstack([(1-alpha)*weights, [alpha]])
                pw2={"prior_weights":w2,"prior_support":support2}
                pw0val = pwval(pw2, xs)
                return kinf_multi(pw1val, pw0val)
            alpha = minimize_scalar(obj, bounds=(0,1), method="bounded").x

        # Try to merge if already exist in support (with slack)
        # Otherwise just add to the support
        D = np.linalg.norm(qopt - support, axis = 1)
        dmin = np.min(D)
        if dmin < tol:
            weights = (1-alpha) * weights
            weights[np.argmin(D)] += alpha
        else:
            support = np.vstack([support, [qopt]])
            weights = np.hstack([(1-alpha)*weights, [alpha]])

        # Trim weights very close to zero
        id_del = []
        while np.min(weights) < tol:
            idmin = np.argmin(weights)
            weights = np.delete(weights, idmin)
            support = np.delete(support, idmin, axis=0)

        pw["prior_support"] = support
        pw["prior_weights"] = weights/np.sum(weights)

        # Stopping condition: mean of distribution did not change too much
        qnew = np.average(pw["prior_support"], 
                          weights=pw["prior_weights"], axis=0)
        if np.linalg.norm(qnew - qold)/np.linalg.norm(qold) < tol:
            break
        qold = qnew
        if (t == maxiter-1) and DEBUG:
            print("Warning: ran out of iterations.")
            break

    if DEBUG:
        print(f"FW finished in {t} steps")

    return pw

