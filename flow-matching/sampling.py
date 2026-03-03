import torch
import numpy as np
from scipy import integrate
from models import utils as mutils
from models.utils import from_flattened_numpy, to_flattened_numpy


def get_sampling_fn(config, sde, shape, inverse_scaler, eps, classifier=None, target_attr_idx=None, guidance_scale=0.0):
    """Create a sampling function."""
    sampler_name = config.sampling.method
    
    if sampler_name.lower() == 'rectified_flow':
        sampling_fn = get_rectified_flow_sampler(
            sde=sde, shape=shape, inverse_scaler=inverse_scaler, device=config.device,
            classifier=classifier, target_attr_idx=target_attr_idx, guidance_scale=guidance_scale
        )
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn


def get_rectified_flow_sampler(sde, shape, inverse_scaler, device='cuda', classifier=None, target_attr_idx=None, guidance_scale=0.0):
    """Get rectified flow sampler."""
  
    def euler_sampler(model, z=None):
        """The probability flow ODE sampler with simple Euler discretization and Guidance."""
        if z is None:
            z0 = sde.get_z0(torch.zeros(shape, device=device), train=False).to(device)
            x = z0.detach().clone()
        else:
            x = z

        model_fn = mutils.get_model_fn(model, train=False) 

        dt = 1. / sde.sample_N
        eps = 1e-3 

        for i in range(sde.sample_N):
            num_t = i / sde.sample_N * (sde.T - eps) + eps
            t = torch.ones(shape[0], device=device) * num_t
            x = x.detach().requires_grad_(True)
            
            with torch.no_grad():
                pred = model_fn(x, t * 999) 

            if classifier is not None and np.abs(guidance_scale) > 0.0 and target_attr_idx is not None:

                with torch.enable_grad():
                    logits = classifier(x, t)   
                    score = logits[:, target_attr_idx].sum()
                    grad = torch.autograd.grad(score, x)[0]
                
                grad_norm = torch.linalg.norm(grad.reshape(grad.shape[0], -1), dim=-1).view(-1, 1, 1, 1)
                grad_normalized = grad / (grad_norm + 1e-8)
                
                scheduled_scale = guidance_scale * (1 - num_t)   #(np.sin(np.pi * num_t))
                
                pred = pred + scheduled_scale * grad_normalized
                
            x = x.detach()
            pred = pred.detach()
            #sigma_t = sde.sigma_t(num_t)
            #pred_sigma = pred + (sigma_t**2)/(2*(sde.noise_scale**2)*((1.-num_t)**2)) * (0.5 * num_t * (1.-num_t) * pred - 0.5 * (2.-num_t)*x)

            x = x + pred * dt #+ sigma_t * np.sqrt(dt) * torch.randn_like(pred_sigma).to(device)

        x = inverse_scaler(x)
        nfe = sde.sample_N
        return x, nfe
  

    def rk45_sampler(model, z=None):
        """The probability flow ODE sampler with black-box ODE solver."""
        with torch.no_grad():
            rtol = atol = sde.ode_tol
            method = 'RK45'
            eps = 1e-3

            if z is None:
                z0 = sde.get_z0(torch.zeros(shape, device=device), train=False).to(device)
                x = z0.detach().clone()
            else:
                x = z
            
            model_fn = mutils.get_model_fn(model, train=False)

            def ode_func(t, x_np):
                x_tensor = from_flattened_numpy(x_np, shape).to(device).type(torch.float32)
                x_tensor = x_tensor.requires_grad_(True)
                
                vec_t = torch.ones(shape[0], device=device) * t
                
                with torch.no_grad():
                    drift = model_fn(x_tensor, vec_t * 999)
                
                if classifier is not None and guidance_scale > 0.0 and target_attr_idx is not None:
                    with torch.enable_grad():
                        logits = classifier(x_tensor, vec_t)
                        score = logits[:, target_attr_idx].sum()
                        grad = torch.autograd.grad(score, x_tensor)[0]
                    
                    grad_norm = torch.linalg.norm(grad.reshape(grad.shape[0], -1), dim=-1).view(-1, 1, 1, 1)
                    grad_normalized = grad / (grad_norm + 1e-8)
                    
                    scheduled_scale = guidance_scale * (1 - t)   #(np.sin(np.pi * t))
                    drift = drift + scheduled_scale * grad_normalized
                    

                return to_flattened_numpy(drift.detach())

            solution = integrate.solve_ivp(ode_func, (eps, sde.T), to_flattened_numpy(x),
                                            rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev
            x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

            x = inverse_scaler(x)
            
            return x, nfe

    if sde.use_ode_sampler == 'rk45':
        return rk45_sampler
    elif sde.use_ode_sampler == 'euler':
        return euler_sampler
    else:
        assert False, 'Not Implemented!'


def get_rectified_flow_inversion_fn(sde, shape, device='cuda'):
    """Get rectified flow ODE inversion sampler (Image -> Noise)."""
  
    def euler_inverter(model, x_1):
        """Inverts an image x_1 back to its latent noise x_0 using pure ODE."""
        x = x_1.detach().clone()
        model_fn = mutils.get_model_fn(model, train=False) 

        dt = 1. / sde.sample_N
        eps = 1e-3 

        with torch.no_grad():
            for i in range(sde.sample_N, 0, -1):
                num_t = i / sde.sample_N * (sde.T - eps) + eps
                t = torch.ones(shape[0], device=device) * num_t
                
                pred = model_fn(x, t * 999) 
                
                x = x - pred * dt

        x_0 = x
        nfe = sde.sample_N
        return x_0, nfe
        
        
    def rk45_inverter(model, x_1):
        """Inverts an image x_1 back to its latent noise x_0 using black-box ODE solver."""
        rtol = atol = sde.ode_tol
        method = 'RK45'
        eps = 1e-3

        x = x_1.detach().clone()
        model_fn = mutils.get_model_fn(model, train=False)

        def ode_func(t, x_np):
            x_tensor = from_flattened_numpy(x_np, shape).to(device).type(torch.float32)
            vec_t = torch.ones(shape[0], device=device) * t
            
            with torch.no_grad():
                drift = model_fn(x_tensor, vec_t * 999)
            
            return to_flattened_numpy(drift)

        with torch.no_grad():
            solution = integrate.solve_ivp(
                ode_func, 
                (sde.T, eps),
                to_flattened_numpy(x),
                rtol=rtol, 
                atol=atol, 
                method=method
            )
            
        nfe = solution.nfev
        x_0 = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)
        
        return x_0, nfe

    print('Type of Inversion Sampler:', sde.use_ode_sampler)
    if sde.use_ode_sampler == 'rk45':
        return rk45_inverter
    elif sde.use_ode_sampler == 'euler':
        return euler_inverter
    else:
        assert False, 'Not Implemented!'
