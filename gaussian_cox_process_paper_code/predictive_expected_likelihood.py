"""
In this file, we will implement the predictive expected likelihood
idea from \citep{Sellier2023}.
"""

"""
Quoting from the paper:
    For a training set X, and a held-out test set X*, we can derive from equation (1)
    an approximation for the expected predictive log-likelihood

    E[log(p(X*|X)] = E_w[(-\int(w'φ (x) + β)^2dx]
                   + ΣE_w[log(w'φ(x*)+β)^2]
    where $w ~  q(w|X, Θ)$.
The sum of expectations can be approximated using Pochhammer series, which we 
approximate using a lookup table. 


It looks to me like the thing that is being constructed here is highly
                         connected to the form of the model that is being
                         used. In general he is trying to approximate the
                         expected log-likelihood, so if we can get an
                         expression for that, we can just use that directly.

"""
