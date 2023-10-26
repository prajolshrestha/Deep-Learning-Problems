import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8


def generate(prompt: str, uncond_prompt=None, input_image=None, strength=0.8, do_cfg=True, cfg_scale=7.5,
             sampler_name="ddpm", n_inference_steps=50, models={}, seed=None, device=None, idle_device=None, tokenizer=None):
    
    with torch.no_grad():
        # Condition met?
        if not (0 < strength <= 1):
            raise ValueError(' Strength should be between 0 and 1')
        
        # Transfer data to idle_device
        if idle_device:
            to_idle =  lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # Random number generator initialized
        generator = torch.Generator(device= device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        # CLIP MODEL: Contrastive language image pre training --> allows to connect text and images
        clip = models["clip"]
        clip.to(device)

        # If classifer free guidence we combine with prompt and without prompt.
        if do_cfg:
            # For prompt: Convert the prompt into tokens using the tokenizer
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            # (Batch_size, seq_len)
            cond_tokens = torch.tensor(cond_tokens, dtype= torch.long, device= device)
            # (batch_size, seq_len) -> (batch_size, seq_len, dim_embed)
            cond_tokens = clip(cond_tokens) # (1,77,768)

            # For unconditional prompt/ negative prompt/ empty string: Convert prompt into tokens using the tokenizer
            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length=77).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device= device)
            # (batch_size, seq_len) -> (batch_size, seq_len, dim_embed)
            uncond_tokens = clip(uncond_tokens) # (1,77,768)

            # Combine conditional tokens and unconditional tokens
            # (batch_size, seq_len, dim_embed) = (2,77,768)
            context = torch.cat([cond_tokens, uncond_tokens])

        else:
            # Use only prompt:
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device= device)
            # (1,77,768)
            context = clip(tokens)
        to_idle(clip)

        # DDPM scheduler:
        if sampler_name == 'ddpm':
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps) # 1000, 980, 960, 940 ... 0 for 50 steps
        else:
            raise ValueError(f"Unknown Sampler {sampler_name}")
        
        latents_shape = (1,4,LATENTS_HEIGHT, LATENTS_WIDTH)

        # For image to image architecture
        if input_image:
            # VAE encoder
            encoder = models["encoder"]
            encoder.to(device)

            # input image
            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            # (height, width, channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            input_image_tensor = rescale(input_image_tensor, (0,255), (-1,1))
            # (height, width, channel) -> (batch_size, height, width, channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (batch_size, height, width, channel) -> (batch_size, channel, height, width)
            input_image_tensor = torch.permute(0,3,1,2)

            # Noise
            encoder_noise = torch.randn(latents_shape, generator= generator, device = device)

            # Run the image through the encoder of VAE 
            latents = encoder(input_image_tensor, encoder_noise)

            # add noise according to strength defined for Sampler 
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        
        # For text-to-image architecture
        else:
            latents = torch.randn(latents_shape, generator= generator, device= device)

        # UNET
        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps) # progress bar
        for i, timestep in enumerate(timesteps):
            # (1,320)
            time_embedding = get_time_embedding(timestep).to(device)

            # (batch_size, 4, latents_height, latents_width)
            model_input = latents

            if do_cfg:
                # (batch_size, 4, latents_height, latents_width) -> (2*batch_size, 4, latents_height, latents_width)
                model_input = model_input.repeat(2,1,1,1)
            
            # model_output is the predicted noise by UNET
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
            
            # Remove noise predicted by UNET
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        # VAE decoder
        decoder = models["decoder"]
        decoder.to(device)

        images = decoder(latents)

        to_idle(decoder)

        images = rescale(images, (-1, 1), (0,255), clamp=True)
        # (batch_size, channel, height, width) -> (batch_size, height, width, channel)
        images = images.permute(0,2,3,1)
        images = images.to("cpu", torch.uint8).numpy()

        return images[0]
    

def rescale(x, old_range, new_range, clamp=False):

    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    
    return x

def get_time_embedding(timestep):
    # (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end = 160, dtype= torch.float32) / 160)
    # (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    
    # (1,320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


                


 