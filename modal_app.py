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
        print("Repository directory exists, removing it to ensure fresh clone.")
        subprocess.run(
            "rm -rf /checkpoints/RS-STE",
            shell=True,
            check=True
        )

    # Clone the repository
    subprocess.run(
        "git clone https://github.com/xunmengshe2x/rs-ste",
        shell=True,
        check=True,
        cwd="/checkpoints"
    )
    
    # Create the logger functions directly in the repository
    with open("/checkpoints/RS-STE/model/utils.py", "r") as f:
        utils_content = f.read()
    
    # Check if get_logger is already in the file
    if "def get_logger" not in utils_content:
        # Add the logger functions to utils.py
        with open("/checkpoints/RS-STE/model/utils.py", "w") as f:
            f.write(utils_content.replace("import torch\nimport torch.nn as nn", "import torch\nimport torch.nn as nn\nimport logging\nimport os\n\ndef get_logger(name, rank=0):\n    \"\"\"\n    Create a logger for the specified name.\n    \n    Args:\n        name (str): Logger name\n        rank (int, optional): Process rank for distributed training. Defaults to 0.\n        \n    Returns:\n        logging.Logger: Configured logger instance\n    \"\"\"\n    logger = logging.getLogger(name)\n    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)\n    \n    # Create console handler\n    ch = logging.StreamHandler()\n    ch.setLevel(logging.INFO if rank == 0 else logging.WARNING)\n    \n    # Create formatter\n    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')\n    ch.setFormatter(formatter)\n    \n    # Add handler to logger\n    logger.addHandler(ch)\n    \n    return logger\n\ndef log_config(config, logger=None):\n    \"\"\"\n    Log configuration parameters.\n    \n    Args:\n        config: Configuration object or dictionary\n        logger (logging.Logger, optional): Logger to use. If None, print to stdout.\n    \"\"\"\n    config_str = str(config)\n    if logger is not None:\n        logger.info(f\"Configuration:\\n{config_str}\")\n    else:\n        print(f\"Configuration:\\n{config_str}\")\n"))
    
    # Verify the file was updated
    with open("/checkpoints/RS-STE/model/utils.py", "r") as f:
        updated_content = f.read()
        if "def get_logger" in updated_content and "def log_config" in updated_content:
            print("Successfully added logger functions to utils.py")
        else:
            print("WARNING: Failed to add logger functions to utils.py")
    
    print("Repository downloaded and patched successfully.")
    return True

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

    # Handle input image from URL
    local_image_path = os.path.join(inputs_dir, "input_image.png")
    urllib.request.urlretrieve(image_url, local_image_path)

    # Set up output path
    outputs_dir = "/checkpoints/outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    output_path = os.path.join(outputs_dir, "output.png")

    # Add the cloned repository to the Python path
    sys.path.append("/checkpoints/RS-STE")

    # Define the logger functions directly in case import fails
    def direct_get_logger(name, rank=0):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO if rank == 0 else logging.WARNING)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger
    
    def direct_log_config(config, logger=None):
        config_str = str(config)
        if logger is not None:
            logger.info(f"Configuration:\n{config_str}")
        else:
            print(f"Configuration:\n{config_str}")
    
    # Monkey patch the model.utils module if needed
    try:
        import model.utils
        if not hasattr(model.utils, 'get_logger'):
            logger.info("Monkey patching model.utils with get_logger")
            model.utils.get_logger = direct_get_logger
        if not hasattr(model.utils, 'log_config'):
            logger.info("Monkey patching model.utils with log_config")
            model.utils.log_config = direct_log_config
    except ImportError:
        logger.warning("Could not import model.utils for monkey patching")

    # Import necessary modules from RS-STE
    try:
        from main import instantiate_from_config, get_obj_from_str
    except ImportError as e:
        if "get_logger" in str(e):
            # Create a temporary main.py with the required functions
            logger.info("Creating temporary main.py with get_logger")
            with open("/checkpoints/RS-STE/main.py", "r") as f:
                main_content = f.read()
            
            # Replace the import line
            patched_content = main_content.replace(
                "from model.utils import get_logger, log_config",
                "# Direct implementation of get_logger and log_config\n"
                "def get_logger(name, rank=0):\n"
                "    import logging\n"
                "    logger = logging.getLogger(name)\n"
                "    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)\n"
                "    ch = logging.StreamHandler()\n"
                "    ch.setLevel(logging.INFO if rank == 0 else logging.WARNING)\n"
                "    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')\n"
                "    ch.setFormatter(formatter)\n"
                "    logger.addHandler(ch)\n"
                "    return logger\n\n"
                "def log_config(config, logger=None):\n"
                "    config_str = str(config)\n"
                "    if logger is not None:\n"
                "        logger.info(f\"Configuration:\\n{config_str}\")\n"
                "    else:\n"
                "        print(f\"Configuration:\\n{config_str}\")"
            )
            
            with open("/checkpoints/RS-STE/main.py", "w") as f:
                f.write(patched_content)
            
            # Try importing again
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

    # Read the output file
    with open(output_path, "rb") as f:
        output_data = f.read()

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
    """Web endpoint for RS-STE inference with direct file upload.
    All inference code is inlined to avoid remote function calls and directory issues."""
    import base64
    import os
    import sys
    import torch
    import logging
    import numpy as np
    from PIL import Image
    from omegaconf import OmegaConf

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Parse the request body
    data = await request.json()
    image_base64 = data.get("image_base64")
    text_prompt = data.get("text_prompt")

    if not image_base64 or not text_prompt:
        return {"error": "Missing required parameters: image_base64 and text_prompt"}

    # Ensure models and repository are downloaded
    download_checkpoint.remote()
    download_repository.remote()

    try:
        # Use absolute paths
        inputs_dir = "/checkpoints/inputs"
        os.makedirs(inputs_dir, exist_ok=True)

        # Create a file for the input image
        image_path = os.path.join(inputs_dir, "input_image.png")

        # Decode and save the image
        image_data = base64.b64decode(image_base64)
        with open(image_path, "wb") as temp_file:
            temp_file.write(image_data)
            temp_file.flush()
            os.fsync(temp_file.fileno())  # Ensure data is written to disk
        
        # Verify the file was written
        if not os.path.exists(image_path):
            logger.error(f"Failed to write image file: {image_path}")
            return {"error": "Failed to write image file"}
        
        file_size = os.path.getsize(image_path)
        logger.info(f"Image file written: {image_path}, size: {file_size} bytes")
        logger.info(f"Directory contents: {os.listdir(inputs_dir)}")
        
        # Set up output path
        outputs_dir = "/checkpoints/outputs"
        os.makedirs(outputs_dir, exist_ok=True)
        output_path = os.path.join(outputs_dir, "output.png")

        # Add the cloned repository to the Python path
        sys.path.append("/checkpoints/RS-STE")

        # Define the logger functions directly to avoid import issues
        def get_logger(name, rank=0):
            logger = logging.getLogger(name)
            logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO if rank == 0 else logging.WARNING)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
            return logger
        
        def log_config(config, logger=None):
            config_str = str(config)
            if logger is not None:
                logger.info(f"Configuration:\n{config_str}")
            else:
                print(f"Configuration:\n{config_str}")
        
        # Monkey patch the model.utils module if needed
        try:
            import model.utils
            if not hasattr(model.utils, 'get_logger'):
                logger.info("Monkey patching model.utils with get_logger")
                model.utils.get_logger = get_logger
            if not hasattr(model.utils, 'log_config'):
                logger.info("Monkey patching model.utils with log_config")
                model.utils.log_config = log_config
        except ImportError:
            logger.warning("Could not import model.utils for monkey patching")

        # Import necessary modules from RS-STE
        try:
            # First try to import directly
            try:
                from main import instantiate_from_config, get_obj_from_str
            except ImportError as e:
                if "get_logger" in str(e):
                    # Create a temporary main.py with the required functions
                    logger.info("Creating temporary main.py with get_logger")
                    
                    # Define the functions directly in the main module namespace
                    import sys
                    import types
                    
                    # Create a new module named 'main'
                    main_module = types.ModuleType('main')
                    
                    # Define the functions in the module
                    exec("""
def get_logger(name, rank=0):
    import logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def log_config(config, logger=None):
    config_str = str(config)
    if logger is not None:
        logger.info(f"Configuration:\\n{config_str}")
    else:
        print(f"Configuration:\\n{config_str}")
                    """, main_module.__dict__)
                    
                    # Now try to patch the main.py file
                    with open("/checkpoints/RS-STE/main.py", "r") as f:
                        main_content = f.read()
                    
                    # Replace the import line
                    patched_content = main_content.replace(
                        "from model.utils import get_logger, log_config",
                        "# Direct implementation of get_logger and log_config\n"
                        "def get_logger(name, rank=0):\n"
                        "    import logging\n"
                        "    logger = logging.getLogger(name)\n"
                        "    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)\n"
                        "    ch = logging.StreamHandler()\n"
                        "    ch.setLevel(logging.INFO if rank == 0 else logging.WARNING)\n"
                        "    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')\n"
                        "    ch.setFormatter(formatter)\n"
                        "    logger.addHandler(ch)\n"
                        "    return logger\n\n"
                        "def log_config(config, logger=None):\n"
                        "    config_str = str(config)\n"
                        "    if logger is not None:\n"
                        "        logger.info(f\"Configuration:\\n{config_str}\")\n"
                        "    else:\n"
                        "        print(f\"Configuration:\\n{config_str}\")"
                    )
                    
                    with open("/checkpoints/RS-STE/main.py", "w") as f:
                        f.write(patched_content)
                    
                    # Try importing again
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
            f.write(f"{os.path.basename(image_path)}\t{text_prompt}")
        
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

        # Verify the output file exists
        if not os.path.exists(output_path):
            logger.error(f"Output file does not exist: {output_path}")
            return {"error": "Failed to generate output image"}
        
        # Read the output file
        with open(output_path, "rb") as f:
            output_data = f.read()
        
        # Encode the output as base64
        encoded_output = base64.b64encode(output_data).decode("utf-8")
        
        return {"image_base64": encoded_output}
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": f"Error processing image: {str(e)}"}

# Define a health check endpoint
@app.function()
@modal.fastapi_endpoint(method="GET")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "rs-ste-inference"}
