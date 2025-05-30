import os
import modal
import base64
from fastapi import Request

# Define a custom image with all dependencies
image = modal.Image.debian_slim().pip_install(
    "numpy>=1.21.0",  # Updated to be compatible with Python 3.10
    "Pillow==9.5.0",
    "torchmetrics==0.5",
    "albumentations==0.4.3",
    "pytorch-lightning==1.4.2",
    "opencv-python",
    "omegaconf",
    "tensorboard",
    "editdistance",
    "tqdm",
    "torch",
    "torchvision",
    "fastapi[standard]",  # Required for web endpoints
    "pydantic>=2.0.0",    # Explicitly add Pydantic
    "typing-extensions"   # Often needed with Pydantic
)

# Add CUDA support, ffmpeg, wget, and git
image = image.apt_install("ffmpeg", "wget", "git")

# Create a Modal volume to store model files
volume = modal.Volume.from_name("rs-ste-models", create_if_missing=True)

# Create a Modal app
app = modal.App("rs-ste-inference", image=image)

@app.function(
    gpu="A10G",  # You can change this to "T4", "A100", etc. based on your needs
    timeout=600,  # 10-minute timeout
    volumes={"/checkpoints": volume}
)
def download_checkpoint():
    """Download RS-STE checkpoint from Hugging Face to the volume."""
    import os
    import subprocess

    # Create checkpoints directory if it doesn't exist
    os.makedirs("/checkpoints", exist_ok=True)

    # Check if checkpoint is already downloaded
    if os.path.exists("/checkpoints/rsste-finetune.ckpt"):
        print("Checkpoint already downloaded.")
        return

    # Download the checkpoint from Hugging Face
    checkpoint_url = "https://huggingface.co/v4mmko/RS-STE/resolve/main/rsste-finetune.ckpt"
    subprocess.run(f"wget {checkpoint_url} -O /checkpoints/rsste-finetune.ckpt", shell=True, check=True)

    # Debug: Print the contents of the /checkpoints directory
    print(f"Contents of /checkpoints: {os.listdir('/checkpoints')}")
    print("Checkpoint downloaded successfully.")
    return True

@app.function(
    gpu="A10G",  # You can change this to "T4", "A100", etc. based on your needs
    timeout=600,  # 10-minute timeout
    volumes={"/checkpoints": volume}
)
def download_repository():
    """Download the RS-STE repository to the volume."""
    import subprocess
    import os

    # Create repository directory if it doesn't exist
    repo_dir = "/checkpoints/RS-STE"
    if os.path.exists(repo_dir):
        print("Repository already downloaded.")
        return

    # Clone the repository
    subprocess.run(
        "git clone https://github.com/ZhengyaoFang/RS-STE.git",
        shell=True,
        check=True,
        cwd="/checkpoints"
    )

    print("Repository downloaded successfully.")
    return True

@app.function(
    gpu="A10G",  # You can change this to "T4", "A100", etc. based on your needs
    timeout=600,  # 10-minute timeout
    volumes={"/checkpoints": volume}
)
def run_inference(
    image_path: str,
    text_prompt: str,
    output_filename: str = "output.png",
    is_url: bool = True
):
    """Run RS-STE inference with the given image and text prompt."""
    import os
    import sys
    import torch
    import urllib.request
    import logging
    import importlib
    import numpy as np
    from PIL import Image
    from omegaconf import OmegaConf

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create inputs directory if it doesn't exist
    inputs_dir = "/checkpoints/inputs"
    os.makedirs(inputs_dir, exist_ok=True)

    # Handle input image (either from URL or local path)
    local_image_path = os.path.join(inputs_dir, "input_image.png")
    if is_url:
        urllib.request.urlretrieve(image_path, local_image_path)
    else:
        # If it's already a local path, just use it
        local_image_path = image_path

    # Debug: Check if the input image file exists
    if not os.path.exists(local_image_path):
        logger.error(f"Input image file does not exist: {local_image_path}")
        raise FileNotFoundError(f"Input image file does not exist: {local_image_path}")

    # Set up output path
    outputs_dir = "/checkpoints/outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    output_path = os.path.join(outputs_dir, output_filename)

    # Add the cloned repository to the Python path
    sys.path.append("/checkpoints/RS-STE")

    # Import necessary modules from RS-STE
    from main import instantiate_from_config, get_obj_from_str
    
    # Load configs
    decoder_config = OmegaConf.load("/checkpoints/RS-STE/configs/vqgan_decoder.yaml")
    config = OmegaConf.load("/checkpoints/RS-STE/configs/synth_pair.yaml")
    config.model.params.decoder_config = decoder_config.model
    config.model.params.ckpt_path = "/checkpoints/rsste-finetune.ckpt"
    
    # Initialize model
    model = instantiate_from_config(config.model).to('cuda')
    model.eval()
    
    # Create a simple dataset for inference
    from data.dataset import InferenceDataset
    
    # Create a temporary annotation file with the input image and text prompt
    temp_annotation_file = os.path.join(inputs_dir, "temp_annotation.txt")
    with open(temp_annotation_file, "w") as f:
        f.write(f"{os.path.basename(local_image_path)}\t{text_prompt}")
    
    # Create dataset and dataloader
    dataset = InferenceDataset(config.data.validation.params.size, temp_annotation_file)
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Run inference
    with torch.no_grad():
        for batch in dataloader:
            img1 = batch["image1"].to('cuda')
            img1 = img1.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
            if img1.dtype == torch.double:
                img1 = img1.float()
            img1_quant_latent_ = model.conv(img1).flatten(2).permute(0, 2, 1)
            rec2_indices, _ = model.str_converter.encode(batch["rec2"])
            rec2_indices = rec2_indices.to(img1_quant_latent_.device)

            rec2_embd = model.str_embd(rec2_indices)
            rec1_mask = model.masked_rec_embd.weight[0][None, None, :].expand(rec2_indices.shape[0], rec2_indices.shape[1], -1)
            img2_mask = model.masked_img_imbd.weight[0][None, None, :].expand(img1_quant_latent_.shape[0], img1_quant_latent_.shape[1], -1)

            inputs = torch.cat([rec2_embd, img1_quant_latent_, rec1_mask, img2_mask], dim=1)
            embeddings, logits = model.transformer(inputs)
            rec1_indices = torch.topk(logits[:,288:320,:], k=1, dim=-1)[1].view(batch["image1"].shape[0],-1)
            pred_rec = model.str_converter.decode(rec1_indices)
            img2_rec_quant_latent = model.conv_o(embeddings[:,320:576,:].permute(0, 2, 1).contiguous().view(embeddings.shape[0], -1, 8, 32))
            img2_rec = model.decoder.decode(img2_rec_quant_latent)

            # Process and save the output image
            def crop_and_resize(x, size, ori_size, k):
                img = x.detach().cpu()
                img = torch.clamp(img, -1., 1.)
                img = (img + 1.0) / 2.0
                img = img.permute(0, 2, 3, 1).numpy()[k]
                img = (img * 255).astype(np.uint8)
                return img

            for k in range(img2_rec.shape[0]):
                edited_img = crop_and_resize(img2_rec, batch["image1_size"], batch['ori_size'], k)
                Image.fromarray(edited_img).save(output_path)

    # Debug: Check if the output file exists
    if not os.path.exists(output_path):
        logger.error(f"Output file does not exist: {output_path}")
        raise FileNotFoundError(f"Output file does not exist: {output_path}")

    # Read the output file
    with open(output_path, "rb") as f:
        output_data = f.read()

    return output_data

# Define a web endpoint for inference with URL
@app.function(
    gpu="A10G",
    timeout=600,
    volumes={"/checkpoints": volume}
)
@modal.fastapi_endpoint(method="POST")
async def inference_api(request: Request):
    """Web endpoint for RS-STE inference using image URL."""
    import base64

    # Parse the request body
    data = await request.json()
    image_url = data.get("image_url")
    text_prompt = data.get("text_prompt")

    if not image_url or not text_prompt:
        return {"error": "Missing required parameters: image_url and text_prompt"}

    # Ensure models and repository are downloaded
    download_checkpoint.remote()
    download_repository.remote()

    # Run inference
    output_data = run_inference.remote(image_url, text_prompt, is_url=True)

    # Encode the output as base64
    encoded_output = base64.b64encode(output_data).decode("utf-8")

    return {"image_base64": encoded_output}

# Define a web endpoint for inference with base64-encoded file
@app.function(
    gpu="A10G",
    timeout=600,
    volumes={"/checkpoints": volume}
)
@modal.fastapi_endpoint(method="POST")
async def inference_api_with_file(request: Request):
    """Web endpoint for RS-STE inference with direct file upload."""
    import base64
    import os

    # Parse the request body
    data = await request.json()
    image_base64 = data.get("image_base64")
    text_prompt = data.get("text_prompt")

    if not image_base64 or not text_prompt:
        return {"error": "Missing required parameters: image_base64 and text_prompt"}

    # Ensure models and repository are downloaded
    download_checkpoint.remote()
    download_repository.remote()

    # Use absolute paths
    inputs_dir = "/checkpoints/inputs"
    os.makedirs(inputs_dir, exist_ok=True)

    # Create a file for the input image
    image_path = os.path.join(inputs_dir, "input_image.png")

    with open(image_path, "wb") as temp_file:
        temp_file.write(base64.b64decode(image_base64))

    # Run inference
    output_data = run_inference.remote(image_path, text_prompt, is_url=False)

    # Encode the output as base64
    encoded_output = base64.b64encode(output_data).decode("utf-8")

    return {"image_base64": encoded_output}

# Define a health check endpoint
@app.function()
@modal.fastapi_endpoint(method="GET")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "rs-ste-inference"}
