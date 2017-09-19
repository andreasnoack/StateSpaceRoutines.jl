# Tempered Particle Filter Documentation

## Introduction
- The tempered particle filter is a particle filtering method which can approximate the log-likelihood value
implied by a general (potentially non-linear) state space system. The filter was introduced by Ed Herbst and
Frank Schorfheide in a recent paper (cited in the [README](https://github.com/FRBNY-DSGE/StateSpaceRoutines.jl)
in the root directory of the StateSpaceRoutines repository).

## Inputs to the function
- The inputs for `tempered_particle_filter` are as follows:
    + The transition and measurement equations: `Φ`,and `Ψ`, which are instances of the `Function` type
    + The shock and measurement error distributions: `F_ϵ`, `F_u`, which are instances of the `Distribution` type
    + A matrix of initial draws (number of states by number of particles): `s_init`.
    These draws can be chosen by the user or can be generated by using the `initialize_state_draws` function,
    which iterates the states forward using `Φ` and `F_ϵ` and generates draws after accounting for an initial
    burn-in period. The sample is also thinned to prevent serial correlation.
    + The tuning hyperparameters (like the number of particles) are set by default but can be specified by
    keyword arguments in the function call. Detailed documentation of these can be found by reading the
    docstring, i.e. calling `?tempered_particle_filter` in the Julia REPL after loading the StateSpaceRoutines
    package.

## General State Space System
```
s_{t+1} = Φ(s_t, ϵ_t)        (transition equation)
y_t     = Ψ(s_t, u_t)        (measurement equation)

ϵ_t ∼ F_ϵ(∙; θ)
u_t ∼ F_u(∙; θ), where F_u is N(0, HH), where HH is the variance matrix of the i.i.d measurement error
Cov(ϵ_t, u_t) = 0
```

## Example: A Linear State Space System derived from a basic DSGE model
- The tempered_particle_filter_ex.jl example file demonstrates the functionality of our Julia implementation of
the tempered particle filter on a linear state space system obtained from solving a basic New Keynesian DSGE
model with Gaussian shocks.

- The example file first loads in a dataset, us.txt, which is an 80-period dataset from 1983-Q1 to 2002-Q4
provided by Herbst and Schorfheide in their [2017 CEF Workshop](https://web.sas.upenn.edu/schorf/cef-2017-herbst-schorfheide-workshop/),
located in the Practical Exercises/MATLAB-PF zip file.

- The solution to the model is then computed using the compute_system function, which is a wrapper script for
setting up and solving the linear rational expectations model using Chris Sims' gensys algorithm. The system
matrices in this example case define the linear transition and measurement equations `Φ` and `Ψ`, and the shock distribution
`F_ϵ` in the example is a multivariate normal distribution (but as mentioned previously, these equations can be generally
non-linear and the distribution generally non-Gaussian).

- We construct a dictionary with the hyperparameters for readability purposes; however these specifications can
also be directly entered into the tempered_particle_filter function as keyword arguments, as opposed to splatting
(`...`) a dictionary into the keyword arguments section of the function call.

- The initial states are drawn from a multivariate normal distribution centered at a prior mean `s0` and
solving the discrete-time Lyapunov equation for the variance-covariance matrix of the states, `P0`. The
`initialize_state_draws` function can be used instead to produce equivalent results.

- When executing the filter, the default return value is the approximated log-likelihood value; however, if the
keyword argument `allout` is set to be true, then the filter will also return the marginal likelihoods at each
period as well as the amount of time it took each marginal likelihood to be calculated.