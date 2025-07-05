from contextlib import contextmanager

import numpy as np
import torch
from torch.autograd.functional import jacobian

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.models.components.sdes import VEReverseSDE, ReverseODE
from dem.utils.data_utils import remove_mean


@contextmanager
def conditional_no_grad(condition):
    if condition:
        with torch.no_grad():
            yield
    else:
        yield


def grad_E(x, energy_function):
    with torch.enable_grad():
        x = x.requires_grad_()
        return torch.autograd.grad(torch.sum(energy_function(x)), x)[0].detach()


def negative_time_descent(x, energy_function, num_steps, dt=1e-4, clipper=None):
    samples = []
    for _ in range(num_steps):
        drift = grad_E(x, energy_function)

        if clipper is not None:
            drift = clipper.clip_scores(drift)

        x = x + drift * dt

        if energy_function.is_molecule:
            x = remove_mean(x, energy_function.n_particles, energy_function.n_spatial_dim)

        samples.append(x)
    return torch.stack(samples)

def divergence(t, x, vectorfield):
    # compute the divergence of the flow field
    def _func_sum(t_in, x_in):
        return vectorfield(t_in, x_in).sum(dim=0) # sum over the batches
    out= jacobian(_func_sum, (t, x),create_graph =True,vectorize=True)
    spatial_grad = out[1]
    div = spatial_grad.diagonal(offset=0, dim1=-1, dim2=-3).sum(-1)
    return div

def compute_score_and_RHS(x, t, schedule, target, prior, clipper):
    outf, doutf = schedule(t)
    x.requires_grad_(True)
    logtarget = target(x)
    dlogtarget = torch.vmap(torch.func.grad(lambda x: target(x)))(x)
    logprior = prior.log_prob(x)
    dlogprior = torch.autograd.grad(logprior.sum(), x, create_graph=True)[0]
    score = outf * dlogtarget + (1-outf) * dlogprior
    score = clipper.clip_scores(score)
    RHS = - (doutf * logtarget  - doutf * logprior)
    return score.detach(), RHS.detach()

def euler_step(ode: ReverseODE, t: torch.Tensor, x: torch.Tensor, dt: float, 
               compute_weight=False, schedule=None, target=None, prior=None,
               clipper=None, w=None):
 
    def compute_is_weight(x , w, t, ode):
        score, RHS = compute_score_and_RHS(x, t, schedule, target, prior, clipper)
        div = divergence(t, x, ode.f)
        vel = ode.f(t, x)
        LHS = div + (score*vel).sum(axis=1)
        dw  = (LHS - (RHS.flatten()))
        return vel.detach(), dw.detach() 
    
    if compute_weight:
        vel , dw = compute_is_weight(x, w, t, ode)
        drift = vel*dt
        return (x + drift).detach(), (w + dw*dt).detach()
    else:
        drift = ode.f(t, x) * dt
        return (x + drift).detach(), torch.zeros(x.shape[0]).to(x.device)

def euler_maruyama_step(
    sde: VEReverseSDE, t: torch.Tensor, x: torch.Tensor, dt: float, diffusion_scale=1.0
):
    # Calculate drift and diffusion terms
    drift = sde.f(t, x) * dt
    diffusion = diffusion_scale * sde.g(t, x) * np.sqrt(dt) * torch.randn_like(x)

    # Update the state
    x_next = x + drift + diffusion
    return x_next, drift


def integrate_pfode(
    sde: VEReverseSDE,
    x0: torch.Tensor,
    num_integration_steps: int,
    reverse_time: bool = True,
):
    start_time = 1.0 if reverse_time else 0.0
    end_time = 1.0 - start_time

    times = torch.linspace(start_time, end_time, num_integration_steps + 1, device=x0.device)[:-1]

    x = x0
    samples = []
    with torch.no_grad():
        for t in times:
            x, f = euler_maruyama_step(sde, t, x, 1 / num_integration_steps)
            samples.append(x)

    return torch.stack(samples)


def integrate_sde(
    sde: VEReverseSDE,
    x0: torch.Tensor,
    num_integration_steps: int,
    energy_function: BaseEnergyFunction,
    reverse_time: bool = True,
    diffusion_scale=1.0,
    no_grad=True,
    time_range=1.0,
    negative_time=False,
    num_negative_time_steps=100,
    clipper=None,
):
    start_time = time_range if reverse_time else 0.0
    end_time = time_range - start_time

    times = torch.linspace(start_time, end_time, num_integration_steps + 1, device=x0.device)[:-1]

    x = x0
    samples = []

    with conditional_no_grad(no_grad):
        for t in times:
            x, f = euler_maruyama_step(
                sde, t, x, time_range / num_integration_steps, diffusion_scale
            )
            if energy_function.is_molecule:
                x = remove_mean(x, energy_function.n_particles, energy_function.n_spatial_dim)
            samples.append(x)

    samples = torch.stack(samples)
    if negative_time:
        print("doing negative time descent...")
        samples_langevin = negative_time_descent(
            x, energy_function, num_steps=num_negative_time_steps, clipper=clipper
        )
        samples = torch.concatenate((samples, samples_langevin), axis=0)

    return samples

def integrate_sde_with_weight(
    sde: VEReverseSDE,
    ode: ReverseODE,
    x0: torch.Tensor,
    w0: torch.Tensor,
    num_integration_steps: int,
    energy_function: BaseEnergyFunction,
    reverse_time: bool = True,
    diffusion_scale=1.0,
    no_grad=True,
    time_range=1.0,
    negative_time=False,
    num_negative_time_steps=100,
    clipper=None,
):
    start_time = time_range if reverse_time else 0.0
    end_time = time_range - start_time

    times = torch.linspace(start_time, end_time, num_integration_steps + 1, device=x0.device)[:-1]

    x = x0
    samples = []
    weights = []
    w = w0
    samples.append(x)
    weights.append(w)
    
    with conditional_no_grad(no_grad):
        for t in times:
            x, f = euler_maruyama_step(
                sde, t, x, time_range / num_integration_steps, diffusion_scale
            )
            if energy_function.is_molecule:
                x = remove_mean(x, energy_function.n_particles, energy_function.n_spatial_dim)
            samples.append(x)

    samples = torch.stack(samples)
    if negative_time:
        print("doing negative time descent...")
        samples_langevin = negative_time_descent(
            x, energy_function, num_steps=num_negative_time_steps, clipper=clipper
        )
        samples = torch.concatenate((samples, samples_langevin), axis=0)

    return samples

# Compute the Importance Sampling Weight
def compute_importance_sampling_weight(x, energy_function):
    log_target = energy_function.log_prob(x)
    log_prior = energy_function.log_prob(x)
    return torch.exp(log_target - log_prior)

def integrate_ode(
    ode: ReverseODE,
    x0: torch.Tensor,
    dt: float,
    end_time: float,
    energy_function: BaseEnergyFunction,
    prior: None,
    reverse_time: bool = True,
    no_grad=True,
    negative_time=False,
    num_negative_time_steps=100,
    clipper=None,
    compute_weight=False,
    schedule=None,
):
    start_time = 1.0 if reverse_time else 0.0
    delta = end_time - start_time
    num_integration_steps = int(torch.abs(delta) / dt)
    times = torch.linspace(start_time, end_time, num_integration_steps + 1, device=x0.device)[:-1]

    x = x0
    samples = []
    weights = []
    w = torch.zeros(x0.shape[0]).to(x0.device)
    samples.append(x)
    weights.append(w)

    if compute_weight:
        no_grad = False # Enable gradient computation for weight computation

    with conditional_no_grad(no_grad):
        for t in times:
            x, w = euler_step(ode, t, x, dt, compute_weight, schedule, energy_function, prior, clipper, w)
            if energy_function.is_molecule:
                x = remove_mean(x, energy_function.n_particles, energy_function.n_spatial_dim)
            samples.append(x)
            weights.append(w)

    samples = torch.stack(samples)
    weights = torch.stack(weights)
    if negative_time:
        print("doing negative time descent...")
        samples_langevin = negative_time_descent(
            x, energy_function, num_steps=num_negative_time_steps, clipper=clipper
        )
        samples = torch.concatenate((samples, samples_langevin), axis=0)
    

    if compute_weight and weights is not None:
        # Get the final weights for resampling 
        final_weights = weights[-1]
        
        # Check if weights are blowing up (using effective sample size as a metric)
        # Calculate effective sample size
        ess = torch.exp(final_weights).sum()**2 / torch.exp(2 * final_weights).sum()
        ess_ratio = ess / final_weights.shape[0]
        
        # Only resample if effective sample size is above threshold (e.g., 0.5)
        # This prevents resampling when weights are degenerate
        if ess_ratio > 0.8:
            # Convert weights to probabilities (normalize)
            # Add small epsilon to avoid division by zero
            epsilon = 1e-8
            probs = torch.exp(final_weights) + epsilon
            probs = probs / probs.sum()
            
            # Perform importance sampling resampling
            # Sample indices based on weights
            num_samples = final_weights.shape[0]
            indices = torch.multinomial(probs, num_samples, replacement=True)
            
            # Resample the final particles based on the sampled indices
            resampled_samples = samples[-1][indices]
            
            # Update the final sample with resampled particles
            samples[-1] = resampled_samples
            
            # Reset weights for resampled particles (they now have equal weights)
            weights[-1] = torch.zeros_like(final_weights)
        else:
            # If weights are blowing up, keep original samples without resampling
            # This prevents numerical instability
            pass

    return samples, weights
