import os, shutil, json, requests, random, time, runpod
from urllib.parse import urlsplit

import torch
from PIL import Image
import numpy as np

from nodes import load_custom_node
from nodes import NODE_CLASS_MAPPINGS
from comfy_extras import nodes_model_advanced, nodes_custom_sampler, nodes_flux

load_custom_node("/content/ComfyUI/custom_nodes/Comfyui-In-Context-Lora-Utils")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI_essentials")

UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
DualCLIPLoader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
LoraLoader = NODE_CLASS_MAPPINGS["LoraLoader"]()
LoadImage = NODE_CLASS_MAPPINGS["LoadImage"]()
AddMaskForICLora = NODE_CLASS_MAPPINGS["AddMaskForICLora"]()
ModelSamplingFlux = nodes_model_advanced.NODE_CLASS_MAPPINGS["ModelSamplingFlux"]()
CLIPTextEncodeFlux = nodes_flux.NODE_CLASS_MAPPINGS["CLIPTextEncodeFlux"]()
RandomNoise = nodes_custom_sampler.NODE_CLASS_MAPPINGS["RandomNoise"]()
BasicGuider = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicGuider"]()
KSamplerSelect = nodes_custom_sampler.NODE_CLASS_MAPPINGS["KSamplerSelect"]()
BasicScheduler = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicScheduler"]()
SetLatentNoiseMask = NODE_CLASS_MAPPINGS["SetLatentNoiseMask"]()
SamplerCustomAdvanced = nodes_custom_sampler.NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
VAEEncode  = NODE_CLASS_MAPPINGS["VAEEncode"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
ImageCrop = NODE_CLASS_MAPPINGS["ImageCrop+"]()

with torch.inference_mode():
    unet = UNETLoader.load_unet("flux1-dev.sft", "default")[0]
    clip = DualCLIPLoader.load_clip("t5xxl_fp16.safetensors", "clip_l.safetensors", "flux")[0]
    vae = VAELoader.load_vae("ae.sft")[0]

def download_file(url, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    file_suffix = os.path.splitext(urlsplit(url).path)[1]
    file_name_with_suffix = file_name + file_suffix
    file_path = os.path.join(save_dir, file_name_with_suffix)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

@torch.inference_mode()
def generate(input):
    values = input["input"]

    input_image = values['input_image_check']
    input_image = download_file(url=input_image, save_dir='/content/ComfyUI/input', file_name='input_image')
    positive_prompt = values['positive_prompt']
    seed = values['seed']
    steps = values['steps']
    guidance = values['guidance']
    sampler_name = values['sampler_name']
    scheduler = values['scheduler']
    max_shift = values['max_shift']
    base_shift = values['base_shift']
    patch_mode = values['patch_mode']
    output_length = values['output_length']
    patch_color = values['patch_color']
    custom_lora1_url = values['custom_lora1_url']
    custom_lora1_file = download_file(url=custom_lora1_url, save_dir='/content/ComfyUI/models/loras', file_name='custom_lora1_file')
    custom_lora1_file = os.path.basename(custom_lora1_file)
    custom_lora1_strength_model = values['custom_lora1_strength_model']
    custom_lora1_strength_clip = values['custom_lora1_strength_clip']
    custom_lora2_url = values['custom_lora2_url']
    custom_lora2_file = download_file(url=custom_lora2_url, save_dir='/content/ComfyUI/models/loras', file_name='custom_lora2_file')
    custom_lora2_file = os.path.basename(custom_lora2_file)
    custom_lora2_strength_model = values['custom_lora2_strength_model']
    custom_lora2_strength_clip = values['custom_lora2_strength_clip']

    if seed == 0:
        random.seed(int(time.time()))
        seed = random.randint(0, 18446744073709551615)
    print(seed)

    custom_lora1_unet, custom_lora1_clip = LoraLoader.load_lora(unet, clip, custom_lora1_file, custom_lora1_strength_model, custom_lora1_strength_clip)
    lora_unet, lora_clip = LoraLoader.load_lora(custom_lora1_unet, custom_lora1_clip, custom_lora2_file, custom_lora2_strength_model, custom_lora2_strength_clip)
    conditioning = CLIPTextEncodeFlux.encode(lora_clip, positive_prompt, positive_prompt, guidance)[0]
    input_image = LoadImage.load_image(input_image)[0]
    return_images, return_masks, min_x, min_y, target_width, target_height, width, height = AddMaskForICLora.add_mask(input_image, patch_mode, output_length, patch_color)
    final_model = ModelSamplingFlux.patch(lora_unet, max_shift, base_shift, width, height)[0]
    noise = RandomNoise.get_noise(seed)[0]
    guider = BasicGuider.get_guider(final_model, conditioning)[0]
    sampler = KSamplerSelect.get_sampler(sampler_name)[0]
    sigmas = BasicScheduler.get_sigmas(lora_unet, scheduler, steps, 1.0)[0]
    latent_image = VAEEncode.encode(vae, return_images)[0]
    latent_image = SetLatentNoiseMask.set_mask(latent_image, return_masks)[0]
    sample, _ = SamplerCustomAdvanced.sample(noise, guider, sampler, sigmas, latent_image)
    decoded = VAEDecode.decode(vae, sample)[0].detach()
    decoded = ImageCrop.execute(decoded, target_width, target_height, "top-left", min_x, min_y)[0]
    image = Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0]).save(f"/content/in-context-lora-{seed}-tost.png")

    result = f"/content/in-context-lora-{seed}-tost.png"
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        with open(result, "rb") as file:
            files = {default_filename: file.read()}
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_url = response.json()['attachments'][0]['url']
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)
        if os.path.exists(input_image):
            os.remove(input_image)

runpod.serverless.start({"handler": generate})