from openai import OpenAI, BadRequestError
import openai
import os
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Response, Depends
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_credentials=True, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

class ImageRequest(BaseModel):
    prompt: str

@app.post("/dalle")
async def generate_image(image_request: ImageRequest):
    try:
        client = OpenAI(api_key='sk-proj-HFvPGP92D4sDyJJgZY6xT3BlbkFJ69FYIoxpLP0nItnzEirx')

        response = client.images.generate(
            model="dall-e-2",
            prompt=image_request.prompt,
            size="256x256",
            quality="standard",
            n=1
        )

        image_url = response.data[0].url
        return JSONResponse(content={"image_url": image_url})
    
    except BadRequestError as e:
        return JSONResponse(status_code=400, content={"detail": f"Bad request: {e}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Internal server error: {e}"})





from auth_token import AuthToken
from torch import autocast
from diffusers import StableDiffusionPipeline
from io import BytesIO
import base64 

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id,  use_auth_token=AuthToken, safety_checker=None)
#device = "cuda"
#pipe.to(device)
pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing("max")


@app.get("/")
def generate(prompt: str): 
    #guidance_scale można zmieniać

    image = pipe(prompt,guidance_scale=8.0).images[0]

    image.save("testimage.png")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    imgstr = base64.b64encode(buffer.getvalue())

    return Response(content=imgstr, media_type="image/png")














import torch
import torchvision
import argparse
import yaml
import os
import sys
sys.path.append('E:\StableDiffusion-PyTorch-main')
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





def sample(model, scheduler, train_config, diffusion_model_config, autoencoder_model_config, diffusion_config, dataset_config, vae, text_tokenizer, text_model):
    im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
    
    xt = torch.randn((1, autoencoder_model_config['z_channels'], im_size, im_size)).to(device)
    
    text_prompt = ['']
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




class TextToImageRequest(BaseModel):
    prompt: str

def load_config():
    config_path = 'E:\StableDiffusion-PyTorch-main\config\celebhq_text_cond.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_models(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet(im_channels=config['autoencoder_params']['z_channels'], model_config=config['ldm_params']).to(device)
    vae = VQVAE(im_channels=config['dataset_params']['im_channels'], model_config=config['autoencoder_params']).to(device)
    scheduler = LinearNoiseScheduler(num_timesteps=config['diffusion_params']['num_timesteps'],
                                     beta_start=config['diffusion_params']['beta_start'],
                                     beta_end=config['diffusion_params']['beta_end'])
    model.eval()
    vae.eval()
    return model, vae, scheduler, device

@app.post("/text-to-image")
async def text_to_image_endpoint(request: TextToImageRequest, config: dict = Depends(load_config)):
    model, vae, scheduler, device = load_models(config)
    final_img = generate_image_from_text(model, vae, scheduler, config, request.prompt, device)
    buffer = BytesIO()
    final_img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return {"image_data": img_str}

def generate_image_from_text(model, vae, scheduler, config, prompt, device):
    # Konfiguracja i parametry
    diffusion_config = config['diffusion_params']
    autoencoder_config = config['autoencoder_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)

    # Pobieranie reprezentacji tekstu
    text_tokenizer, text_model = get_text_representation(prompt, condition_config['text_embed_model'], device)
    text_prompt_embed = get_text_representation([prompt], text_tokenizer, text_model, device)

    # Inicjalizacja szumu
    im_size = dataset_config['im_size'] // (2 ** sum(autoencoder_config['down_sample']))
    xt = torch.randn((1, autoencoder_config['z_channels'], im_size, im_size)).to(device)

    # Pętla dyfuzji
    for i in tqdm(reversed(range(diffusion_config['num_timesteps'])), desc='Generating image'):
        t = (torch.ones((xt.shape[0],)) * i).long().to(device)
        noise_pred_cond = model(xt, {'t': t, 'text': text_prompt_embed})
        if config['cf_guidance_scale'] > 1:
            noise_pred_uncond = model(xt, {'t': t, 'text': torch.zeros_like(text_prompt_embed)})
            noise_pred = noise_pred_uncond + config['cf_guidance_scale'] * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = noise_pred_cond

        xt, _ = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

    # Dekodowanie VQVAE do obrazu finalnego
    final_img = vae.decode(xt)
    final_img = torch.clamp(final_img, -1., 1.)  # Normalizacja obrazu do zakresu [0,1]
    final_img = (final_img + 1) / 2  # Dopasowanie do formatu PIL

    # Tworzenie siatki obrazów (jeśli jest więcej niż jeden obraz)
    grid = make_grid(final_img, nrow=1)
    return torchvision.transforms.ToPILImage()(grid)
