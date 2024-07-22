import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.config_utils import *
from utils.text_utils import *
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_clip_score(images, text_prompts, device):
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    inputs = clip_processor(text=text_prompts, images=images, return_tensors="pt", padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    outputs = clip_model(**inputs)
    text_features = outputs.text_embeds
    image_features = outputs.image_embeds
    
    text_features = F.normalize(text_features, p=2, dim=-1)
    image_features = F.normalize(image_features, p=2, dim=-1)
    clip_score = (text_features * image_features).sum(dim=1)
    
    return clip_score.mean().item()



def sample(model, scheduler, train_config, diffusion_model_config, autoencoder_model_config, diffusion_config, dataset_config, vae, text_tokenizer, text_model):
    im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
    
    xt = torch.randn((1, autoencoder_model_config['z_channels'], im_size, im_size)).to(device)
    
    text_prompt = ['This woman has arched eyebrows, big nose, high cheekbones, receding hairline, and straight hair and is wearing earrings, and lipstick. She is smiling.']
    empty_prompt = ['']
    text_prompt_embed = get_text_representation(text_prompt, text_tokenizer, text_model, device)
    empty_text_embed = get_text_representation(empty_prompt, text_tokenizer, text_model, device)
    
    uncond_input = {'text': empty_text_embed}
    cond_input = {'text': text_prompt_embed}
    
    cf_guidance_scale = get_config_value(train_config, 'cf_guidance_scale', 8.0)
    
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        t = (torch.ones((xt.shape[0],)) * i).long().to(device)
        noise_pred_cond = model(xt, t, cond_input)
        
        if cf_guidance_scale > 1:
            noise_pred_uncond = model(xt, t, uncond_input)
            noise_pred = noise_pred_uncond + cf_guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = noise_pred_cond
        
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
        
        if i == 0:
            final_img = vae.decode(xt)
        
    final_img = torch.clamp(final_img, -1., 1.).detach().cpu()
    final_img = (final_img + 1) / 2
    grid = make_grid(final_img, nrow=1)
    img = torchvision.transforms.ToPILImage()(grid)
    
    if final_img.dim() == 3:
        final_img = final_img.unsqueeze(0)  # Add batch dimension only if it's not already a 4D tensor

    # Now final_img should be [1, C, H, W]
    clip_score = calculate_clip_score(images=final_img, text_prompts=text_prompt, device=device)
    print(f"Final CLIP Score: {clip_score}")
    
    save_path = os.path.join(train_config['task_name'], 'cond_text_samples', 'final_image.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)
    img.close()
    
def infer(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    assert condition_config is not None, "No conditioning config found"
    condition_types = get_config_value(condition_config, 'condition_types', [])
    assert 'text' in condition_types, "No text condition found in config"
    validate_text_config(condition_config)
    
    with torch.no_grad():
        text_tokenizer, text_model = get_tokenizer_and_model(condition_config['text_condition_config']['text_embed_model'], device=device)
    
    model = Unet(im_channels=autoencoder_model_config['z_channels'], model_config=diffusion_model_config).to(device)
    model.eval()
    model_ckpt_path = os.path.join(train_config['task_name'], train_config['ldm_ckpt_name'])
    if os.path.exists(model_ckpt_path):
        print('Loaded unet checkpoint')
        model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
    else:
        raise Exception(f'Model checkpoint {model_ckpt_path} not found')
    
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    
    vae = VQVAE(im_channels=dataset_config['im_channels'], model_config=autoencoder_model_config).to(device)
    vae.eval()
    vae_ckpt_path = os.path.join(train_config['task_name'], train_config['vqvae_autoencoder_ckpt_name'])
    if os.path.exists(vae_ckpt_path):
        print('Loaded vae checkpoint')
        vae.load_state_dict(torch.load(vae_ckpt_path, map_location=device), strict=True)
    else:
        raise Exception(f'VAE checkpoint {vae_ckpt_path} not found')
    
    with torch.no_grad():
        sample(model, scheduler, train_config, diffusion_model_config, autoencoder_model_config, diffusion_config, dataset_config, vae, text_tokenizer, text_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation with only text conditioning')
    parser.add_argument('--config', dest='config_path', default='config/celebhq_text_cond.yaml', type=str)
    args = parser.parse_args()
    infer(args)
