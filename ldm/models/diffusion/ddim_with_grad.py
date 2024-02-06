"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image


from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor


class DDIMSamplerWithGrad(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):

        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.model.module.num_timesteps, verbose=verbose)

        alphas_cumprod = self.model.module.alphas_cumprod
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.module.device)

        self.register_buffer('betas', to_torch(self.model.module.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.module.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)


    def sample(self,
               batch_size,
               shape,
               num_ddim_steps=500,
               src_img=None,
               cond_embed=None,
               uncond_embed=None,
               eta=0.,
               CFG_scale=1.,
               start_zt=None,
               cached_latents=None,
               edit_mask=None,
               num_recursive_steps=1,
               clip_grad=0,
               guidance_weight=1.0,
               guidance_schedule=None,
               log_freq=0,
               results_folder=None,
               guidance_energy=None
            ):

        # Make folders to save stuff in
        if log_freq != 0:
            recon_save_dir = results_folder / 'recons'
            recon_save_dir.mkdir(exist_ok=True)
            flow_save_dir = results_folder / 'flow_viz'
            flow_save_dir.mkdir(exist_ok=True)

        # Get schedule info
        self.make_schedule(ddim_num_steps=num_ddim_steps, ddim_eta=eta, verbose=False)
        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas

        # Get steps info
        timesteps = self.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        # Get shape and device info
        C, H, W = shape
        shape = (batch_size, C, H, W)
        device = self.model.module.betas.device
        b = batch_size

        # Make guidance schedule if we didn't get one
        if guidance_schedule is None:
            guidance_schedule = np.ones(500)

        # Get initial latent if we don't have cached latents (for edit mask copying)
        # We will inject noise (sample from forward process) later on
        if cached_latents is None:
            tgt_z0 = self.model.module.encode_first_stage(src_img).mean * 0.18215

        # Init noisy latent
        if start_zt is None:
            noisy_latent = torch.randn(shape, device=device)
            start_zt = noisy_latent
        else:
            noisy_latent = start_zt

        # Freeze model
        for param in self.model.module.first_stage_model.parameters():
            param.requires_grad = False

        # Track stuff 
        losses = []
        losses_flow = []
        losses_color = []
        noise_norms = []
        guidance_norms = []

        # DDIM sampling loop
        for i, step in enumerate(iterator):
            # Bookkeeping
            index = total_steps - 1 - i
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            b, *_, device = *noisy_latent.shape, noisy_latent.device

            # Get variance schedule params
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
            beta_t = a_t / a_prev

            # For each step, do N extra "recursive" steps
            for j in range(num_recursive_steps):

                # Replace latents outside of edit mask, similar to RePaint
                if cached_latents is not None:
                    gt_latent = cached_latents[i]
                else:
                    noise = noise_like(noisy_latent.shape, device, False)
                    gt_latent = a_t.sqrt() * tgt_z0 + (1 - a_t).sqrt() * noise
                noisy_latent[edit_mask] = gt_latent[edit_mask]

                # Set up grad (for differentiating guidance energy)
                torch.set_grad_enabled(True)
                noisy_latent_grad = noisy_latent.detach().requires_grad_(True)

                # Get CFG noise estimates
                x_in = torch.cat([noisy_latent_grad] * 2)
                t_in = torch.cat([ts] * 2)
                c_in = torch.cat([uncond_embed, cond_embed])
                e_t_uncond, e_t = self.model.module.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + CFG_scale * (e_t - e_t_uncond)

                # Get one step approximation
                pred_x0 = (noisy_latent_grad - sqrt_one_minus_at * e_t) / a_t.sqrt()
                recons_image = self.model.module.decode_first_stage_with_grad(pred_x0)

                # Get gradient of energy function
                energy, info_loss = guidance_energy(recons_image, src_img)
                grad = torch.autograd.grad(energy, noisy_latent_grad)[0]
                grad = -grad * guidance_weight * guidance_schedule[i]

                # Clip gradient
                if clip_grad != 0:
                    grad_norm = torch.linalg.norm(sqrt_one_minus_at * grad.detach())
                    if grad_norm > clip_grad:
                        grad = grad / grad_norm * clip_grad

                # Update noise estimate with guidance gradiaent
                    # TODO: This may be a bug, try not multiplying here. But need new hyperparams probs
                e_t = e_t - sqrt_one_minus_at * grad.detach()   
                noisy_latent_grad = noisy_latent_grad.requires_grad_(False)

                # Logging
                losses.append(energy.item())
                losses_flow.append(info_loss['flow_loss'])
                losses_color.append(info_loss['color_loss'])
                noise_norms.append(torch.linalg.norm(e_t).item())
                guidance_norms.append(torch.linalg.norm(sqrt_one_minus_at * grad.detach()).item())

                # Save images
                if log_freq != 0 and i % log_freq == 0 and j == 0:
                    # Save reconstruction
                    temp = (recons_image + 1) * 0.5
                    save_image(temp, recon_save_dir / f'xt.{i:05}.png')

                    # Save image of flow
                    info_loss['flow_im'].save(flow_save_dir / f'flow.{i:05}.png')

                del noisy_latent_grad, pred_x0, recons_image, grad, x_in

                torch.set_grad_enabled(False)

                # DDIM step
                with torch.no_grad():
                    # current prediction for x_0
                    pred_x0 = (noisy_latent - sqrt_one_minus_at * e_t) / a_t.sqrt()

                    # direction pointing to x_t
                    dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t

                    # random noise
                    noise = sigma_t * noise_like(noisy_latent.shape, device, False)

                    # DDIM step, get prev latent z_{t-1}
                    noisy_latent_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

                    # Inject noise (sample forward process) for recursive denoising
                    recur_noise = noise_like(noisy_latent.shape, device, False)
                    noisy_latent = beta_t.sqrt() * noisy_latent_prev + (1 - beta_t).sqrt() * recur_noise

                    del pred_x0, dir_xt, noise

            # When done making recursive steps, set current latent to t-1 latent
            noisy_latent = noisy_latent_prev


        # Make info dict to output
        info = {
                    'losses': np.array(losses),
                    'losses_flow': np.array(losses_flow),
                    'losses_color': np.array(losses_color),
                    'noise_norms': np.array(noise_norms),
                    'guidance_norms': np.array(guidance_norms)
               }

        return noisy_latent, start_zt, info
