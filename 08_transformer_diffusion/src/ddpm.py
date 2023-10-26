import torch
import numpy as np

class DDPMSampler:

    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start: float= 0.00085 , beta_end: float= 0.0120):
        """
            Initialize beta (variance of gaussian noise) required for closed form formula of diffusion process/Forward process.
        """

        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32)**2
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, 0) # [alpha_0, alpha_0*alpha_1, alpha_0*alpha_1*alpha_2, ...]
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy()) # 999, 998, ... 0 = 1000 steps

    def set_inference_timesteps(self, num_inference_steps=50):
        """
            It calculates timesteps: 999, 999-20, 999-40, ....0 = 50 steps
        """
        self.num_inference_steps = num_inference_steps
        # 999, 999-20, 999-40, ....0 = 50 steps
        step_ratio = self.num_training_steps // self.num_inference_steps # 20
        timesteps = (np.arange(0, num_inference_steps)*step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = timestep - (self.num_training_steps // self.num_inference_steps) # eg. if timestep = 999, prev_t = 999 - 20

        return prev_t
    
    def _get_variance(self, timestep: int) -> int:

        prev_t = self._get_previous_timestep(timestep)

        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        curernt_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * curernt_beta_t

        variance = torch.clamp(variance, min=1e20)

        return variance

    def set_strength(self, strength=1):
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step



    def step(self, timestep:int, latents: torch.Tensor, model_output: torch.Tensor):
        """
            Reverse Process: amount of noise predicted my UNET (ie, model_output) is removed here.

            timestep: at which timestep we should remove the noise?
            latents: output of VAE encoder
            model_output: predicted amount of noise by UNET
        """
        # Load all required data
        t = timestep
        prev_t = self._get_previous_timestep(t)

        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one 
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        
        ## MEAN 
        # Commpute the predicted original sample (x_0) using formule (15) of DDPM paper
        pred_original_sample = (latents - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

        # compute the cofficients for pred_original_sample(x_0) and current sample (x_t)
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t
        pred_current_sample_coeff = (current_alpha_t ** 0.5 * beta_prod_t_prev) / beta_prod_t

        # Compute the predicted previous sample mean
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + pred_current_sample_coeff * latents 

        ## STD
        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device =device, dtype=model_output.dtype) 
            variance = (self._get_variance(t) ** 0.5) * noise

        # z=N(0,1) -> N(mean, std)=x
        # x = mean + std * z
        pred_prev_sample = pred_prev_sample + variance # z already included (ie, noise)

        return pred_prev_sample


    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        """ 
            Diffusion Process/Forward Process: (closed form formula used)- sample noise from gaussian distribution N(mean, std)

            original_samples: a sample where noise will be added
            timesteps: At what time we want to add noise?
        """

        alpha_cumprod = self.alpha_cumprod.to(device= original_samples.device, dtype= original_samples.dtype)
        timesteps = timesteps.to(device=original_samples.device)

        ## Mean for our Gaussian Distribution
        sqrt_alpha_prod = alpha_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten() # no dimension

        # let's add dimension to sqrt_alpha_prod such that its dimension becomes dimension of original samples
        # This is done for broadcasting purpose when we multiply them.
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        ## Standard Deviation for our Gaussian Distribution
        sqrt_one_minus_alpha_prod = (1 - alpha_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()

        # lets add dimension as before
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # According to the equation (4) of the DDPM paper.
        # z=N(0,1) -> N(mean, std)=x
        # x = mean + std * z
        noise = torch.randn(original_samples.shape, generator = self.generator, device= original_samples.device, dtype= original_samples.dtype)
        noisy_samples = (sqrt_alpha_prod * original_samples) + (sqrt_one_minus_alpha_prod) * noise

        return noisy_samples

