import os
import modal
import base64
import time
import uuid
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
    "typing-extensions",  # Often needed with Pydantic
    "einops",            # Required by model/decoder.py
    "ftfy",              # Common dependency for text processing
    "regex",             # Common dependency for text processing
    "transformers"       # Common dependency for transformer models
)

# Add CUDA support, ffmpeg, wget, and git
image = image.apt_install("ffmpeg", "wget", "git")

# Create a Modal volume to store model files
volume = modal.Volume.from_name("rs-ste-models", create_if_missing=True)

# Create a Modal app
app = modal.App("rs-ste-inference", image=image)

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
    import subprocess
    import time
    import json
    import shutil
    import uuid
    import random

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Parse the request body
    data = await request.json()
    image_base64 = data.get("image_base64")
    text_prompt = data.get("text_prompt")

    if not image_base64 or not text_prompt:
        return {"error": "Missing required parameters: image_base64 and text_prompt"}

    try:
        # Generate a unique deployment ID to avoid caching issues
        deployment_id = str(uuid.uuid4())[:8]
        timestamp = int(time.time())
        random_suffix = random.randint(1000, 9999)
        unique_id = f"{deployment_id}_{timestamp}_{random_suffix}"
        logger.info(f"Generated unique deployment ID: {unique_id}")
        
        # Ensure checkpoint is downloaded
        if not os.path.exists("/checkpoints/rsste-finetune.ckpt"):
            logger.info("Downloading main checkpoint...")
            checkpoint_url = "https://huggingface.co/v4mmko/RS-STE/resolve/main/rsste-finetune.ckpt"
            subprocess.run(f"wget {checkpoint_url} -O /checkpoints/rsste-finetune.ckpt", shell=True, check=True)
            logger.info("Main checkpoint downloaded successfully")

        # Define repository paths
        base_repo_dir = "/checkpoints/RS-STE"
        unique_repo_dir = f"/checkpoints/RS-STE_{unique_id}"
        
        # CACHE BUSTING: Force removal of existing repository
        logger.info("CACHE BUSTING: Forcefully removing any existing repository...")
        
        # First try using Python's shutil
        if os.path.exists(base_repo_dir):
            try:
                logger.info(f"Removing directory with shutil: {base_repo_dir}")
                shutil.rmtree(base_repo_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Error removing directory with shutil: {e}")
        
        # Then use shell commands for more forceful removal
        try:
            logger.info(f"Removing directory with rm -rf: {base_repo_dir}")
            subprocess.run(f"rm -rf {base_repo_dir}", shell=True, check=True)
        except Exception as e:
            logger.warning(f"Error removing directory with rm -rf: {e}")
        
        # Verify removal was successful
        if os.path.exists(base_repo_dir):
            logger.warning(f"Directory still exists after removal attempts: {base_repo_dir}")
            # List any remaining files
            try:
                remaining_files = os.listdir(base_repo_dir)
                logger.warning(f"Remaining files: {remaining_files}")
            except Exception as e:
                logger.warning(f"Error listing remaining files: {e}")
            
            # Try one more extreme removal method
            try:
                logger.info("Attempting extreme removal with find and delete")
                subprocess.run(f"find {base_repo_dir} -delete", shell=True, check=True)
            except Exception as e:
                logger.warning(f"Error with extreme removal: {e}")
        
        # Clone to a unique directory to avoid any caching issues
        logger.info(f"Cloning repository to unique directory: {unique_repo_dir}")
        subprocess.run(
            f"git clone https://github.com/xunmengshe2x/rs-ste {unique_repo_dir}",
            shell=True,
            check=True
        )
        
        # Verify the unique repository was cloned successfully
        if not os.path.exists(unique_repo_dir):
            raise FileNotFoundError(f"Unique repository directory was not created after git clone: {unique_repo_dir}")
        
        # Create a symlink from the base path to the unique path
        logger.info(f"Creating symlink: {base_repo_dir} -> {unique_repo_dir}")
        if os.path.exists(base_repo_dir):
            os.remove(base_repo_dir)
        os.symlink(unique_repo_dir, base_repo_dir)
        
        # Verify the symlink was created successfully
        if not os.path.exists(base_repo_dir):
            raise FileNotFoundError(f"Symlink was not created successfully: {base_repo_dir}")
        
        # List contents to verify
        logger.info(f"Repository contents: {os.listdir(base_repo_dir)}")
        
        # Verify model directory exists
        if not os.path.exists(f"{base_repo_dir}/model"):
            raise FileNotFoundError("Model directory not found in cloned repository")
        
        # Verify utils.py exists
        utils_path = f"{base_repo_dir}/model/utils.py"
        if not os.path.exists(utils_path):
            raise FileNotFoundError(f"utils.py not found at {utils_path}")
        
        # Add logger functions to utils.py
        with open(utils_path, "r") as f:
            utils_content = f.read()
        
        if "def get_logger" not in utils_content:
            logger.info("Adding logger functions to utils.py")
            with open(utils_path, "w") as f:
                f.write(utils_content.replace("import torch\nimport torch.nn as nn", "import torch\nimport torch.nn as nn\nimport logging\nimport os\n\ndef get_logger(name, rank=0):\n    \"\"\"\n    Create a logger for the specified name.\n    \n    Args:\n        name (str): Logger name\n        rank (int, optional): Process rank for distributed training. Defaults to 0.\n        \n    Returns:\n        logging.Logger: Configured logger instance\n    \"\"\"\n    logger = logging.getLogger(name)\n    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)\n    \n    # Create console handler\n    ch = logging.StreamHandler()\n    ch.setLevel(logging.INFO if rank == 0 else logging.WARNING)\n    \n    # Create formatter\n    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')\n    ch.setFormatter(formatter)\n    \n    # Add handler to logger\n    logger.addHandler(ch)\n    \n    return logger\n\ndef log_config(config, logger=None):\n    \"\"\"\n    Log configuration parameters.\n    \n    Args:\n        config: Configuration object or dictionary\n        logger (logging.Logger, optional): Logger to use. If None, print to stdout.\n    \"\"\"\n    config_str = str(config)\n    if logger is not None:\n        logger.info(f\"Configuration:\\n{config_str}\")\n    else:\n        print(f\"Configuration:\\n{config_str}\")\n"))
        
        logger.info("Repository prepared successfully")

        # Check for decoder checkpoint and download if needed
        decoder_ckpt_path = f"{base_repo_dir}/weights/decoder.ckpt"
        decoder_dir = os.path.dirname(decoder_ckpt_path)
        
        # Create weights directory if it doesn't exist
        if not os.path.exists(decoder_dir):
            os.makedirs(decoder_dir, exist_ok=True)
            logger.info(f"Created weights directory: {decoder_dir}")
        
        # Check if decoder checkpoint exists, if not, download it
        if not os.path.exists(decoder_ckpt_path):
            logger.info("Decoder checkpoint not found, downloading...")
            # Try to download from Hugging Face
            decoder_url = "https://huggingface.co/v4mmko/RS-STE/resolve/main/decoder.ckpt"
            try:
                subprocess.run(f"wget {decoder_url} -O {decoder_ckpt_path}", shell=True, check=True)
                logger.info(f"Decoder checkpoint downloaded to {decoder_ckpt_path}")
            except Exception as e:
                logger.warning(f"Failed to download decoder checkpoint: {e}")
                # If download fails, try to use the main checkpoint for both
                logger.info("Using main checkpoint for decoder as fallback")
                subprocess.run(f"cp /checkpoints/rsste-finetune.ckpt {decoder_ckpt_path}", shell=True, check=True)
        
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
        sys.path.append(base_repo_dir)

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
        except ImportError as e:
            logger.warning(f"Could not import model.utils for monkey patching: {e}")
            # Create the module and patch it
            import sys
            import types
            model_module = types.ModuleType('model')
            utils_module = types.ModuleType('model.utils')
            utils_module.get_logger = get_logger
            utils_module.log_config = log_config
            sys.modules['model'] = model_module
            sys.modules['model.utils'] = utils_module
            logger.info("Created and patched model.utils module")

        # Import necessary modules from RS-STE
        try:
            # First try to import directly
            try:
                from main import instantiate_from_config, get_obj_from_str
            except ImportError as e:
                logger.warning(f"Import error: {e}")
                if "get_logger" in str(e):
                    # Create a temporary main.py with the required functions
                    logger.info("Creating temporary main.py with get_logger")
                    
                    # Patch main.py file
                    main_path = f"{base_repo_dir}/main.py"
                    if os.path.exists(main_path):
                        with open(main_path, "r") as f:
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
                        
                        with open(main_path, "w") as f:
                            f.write(patched_content)
                        
                        logger.info("main.py patched successfully")
                    else:
                        logger.error(f"main.py not found at {main_path}")
                    
                    # Try importing again
                    from main import instantiate_from_config, get_obj_from_str
        except Exception as e:
            logger.error(f"Error importing modules: {str(e)}")
            return {"error": f"Error importing modules: {str(e)}"}
        
        # Load configs
        try:
            from omegaconf import OmegaConf
            decoder_config_path = f"{base_repo_dir}/configs/vqgan_decoder.yaml"
            synth_pair_path = f"{base_repo_dir}/configs/synth_pair.yaml"
            
            # List all config files to verify
            config_dir = f"{base_repo_dir}/configs"
            logger.info(f"Config directory contents: {os.listdir(config_dir)}")
            
            if not os.path.exists(decoder_config_path):
                logger.error(f"Decoder config not found at {decoder_config_path}")
                # Try to find any decoder config
                for file in os.listdir(config_dir):
                    if "vqgan" in file.lower() or "decoder" in file.lower():
                        decoder_config_path = os.path.join(config_dir, file)
                        logger.info(f"Found alternative decoder config: {decoder_config_path}")
                        break
                else:
                    return {"error": f"Decoder config not found in {config_dir}"}
            
            if not os.path.exists(synth_pair_path):
                logger.error(f"Synth pair config not found at {synth_pair_path}")
                return {"error": f"Synth pair config not found at {synth_pair_path}"}
            
            # Load and patch configs with absolute paths
            logger.info(f"Loading decoder config from: {decoder_config_path}")
            decoder_config = OmegaConf.load(decoder_config_path)
            logger.info(f"Loading synth pair config from: {synth_pair_path}")
            config = OmegaConf.load(synth_pair_path)
            
            # Log the original configs before patching
            logger.info(f"Original decoder config: {OmegaConf.to_yaml(decoder_config)}")
            logger.info(f"Original synth pair config: {OmegaConf.to_yaml(config)}")
            
            # Patch relative paths in configs to absolute paths
            def fix_paths_in_config(cfg, base_path):
                """Recursively fix relative paths in config to absolute paths"""
                if isinstance(cfg, dict):
                    for key, value in cfg.items():
                        if isinstance(value, (dict, list)):
                            fix_paths_in_config(value, base_path)
                        elif isinstance(value, str) and (
                            "data/" in value or 
                            "weights/" in value or
                            value.endswith(".txt") or 
                            value.endswith(".pkl") or 
                            value.endswith(".yaml") or
                            value.endswith(".ckpt")
                        ):
                            # Check if it's a relative path
                            if not value.startswith("/"):
                                # Make it absolute
                                abs_path = os.path.join(base_path, value)
                                logger.info(f"Converting path: {value} -> {abs_path}")
                                cfg[key] = abs_path
                elif isinstance(cfg, list):
                    for i, item in enumerate(cfg):
                        if isinstance(item, (dict, list)):
                            fix_paths_in_config(item, base_path)
                        elif isinstance(item, str) and (
                            "data/" in item or 
                            "weights/" in item or
                            item.endswith(".txt") or 
                            item.endswith(".pkl") or 
                            item.endswith(".yaml") or
                            item.endswith(".ckpt")
                        ):
                            # Check if it's a relative path
                            if not item.startswith("/"):
                                # Make it absolute
                                abs_path = os.path.join(base_path, item)
                                logger.info(f"Converting path: {item} -> {abs_path}")
                                cfg[i] = abs_path
            
            # Fix paths in both configs
            fix_paths_in_config(decoder_config, base_repo_dir)
            fix_paths_in_config(config, base_repo_dir)
            
            # Specifically check and fix the alphabet path
            if hasattr(config.model.params, 'alphabet') and not config.model.params.alphabet.startswith('/'):
                alphabet_path = config.model.params.alphabet
                abs_alphabet_path = os.path.join(base_repo_dir, alphabet_path)
                logger.info(f"Converting alphabet path: {alphabet_path} -> {abs_alphabet_path}")
                config.model.params.alphabet = abs_alphabet_path
                
                # Verify the alphabet file exists
                if not os.path.exists(abs_alphabet_path):
                    logger.error(f"Alphabet file not found at {abs_alphabet_path}")
                    # List the directory contents to help debug
                    alphabet_dir = os.path.dirname(abs_alphabet_path)
                    if os.path.exists(alphabet_dir):
                        logger.info(f"Contents of alphabet directory: {os.listdir(alphabet_dir)}")
                    else:
                        logger.error(f"Alphabet directory not found: {alphabet_dir}")
                    return {"error": f"Alphabet file not found at {abs_alphabet_path}"}
            
            # CRITICAL FIX: Explicitly set the decoder checkpoint path to absolute path
            # This ensures the nested decoder config has the correct absolute path
            if hasattr(decoder_config.model.params, 'ckpt_path'):
                decoder_ckpt_rel_path = decoder_config.model.params.ckpt_path
                if not decoder_ckpt_rel_path.startswith('/'):
                    decoder_ckpt_abs_path = os.path.join(base_repo_dir, decoder_ckpt_rel_path)
                    logger.info(f"Converting decoder checkpoint path: {decoder_ckpt_rel_path} -> {decoder_ckpt_abs_path}")
                    decoder_config.model.params.ckpt_path = decoder_ckpt_abs_path
            
            # Set decoder config and checkpoint path
            config.model.params.decoder_config = decoder_config.model
            config.model.params.ckpt_path = "/checkpoints/rsste-finetune.ckpt"
            
            # CRITICAL FIX: Directly set the nested decoder checkpoint path to absolute path
            # This ensures the decoder checkpoint path is absolute even if the config patching missed it
            if hasattr(config.model.params.decoder_config.params, 'ckpt_path'):
                nested_ckpt_path = config.model.params.decoder_config.params.ckpt_path
                if not nested_ckpt_path.startswith('/'):
                    abs_nested_ckpt_path = os.path.join(base_repo_dir, nested_ckpt_path)
                    logger.info(f"Converting nested decoder checkpoint path: {nested_ckpt_path} -> {abs_nested_ckpt_path}")
                    config.model.params.decoder_config.params.ckpt_path = abs_nested_ckpt_path
                    
                    # Verify the decoder checkpoint exists at the absolute path
                    if not os.path.exists(abs_nested_ckpt_path):
                        logger.warning(f"Nested decoder checkpoint not found at {abs_nested_ckpt_path}, using main checkpoint as fallback")
                        # Use the main checkpoint as fallback
                        subprocess.run(f"cp /checkpoints/rsste-finetune.ckpt {abs_nested_ckpt_path}", shell=True, check=True)
            
            # Log the patched configs
            logger.info(f"Patched config: {OmegaConf.to_yaml(config)}")
            logger.info(f"Decoder config ckpt_path: {config.model.params.decoder_config.params.ckpt_path}")
            
            # Verify the decoder checkpoint path is absolute
            if not config.model.params.decoder_config.params.ckpt_path.startswith('/'):
                logger.error("Decoder checkpoint path is still relative after patching!")
                # Force it to be absolute as a last resort
                config.model.params.decoder_config.params.ckpt_path = f"{base_repo_dir}/weights/decoder.ckpt"
                logger.info(f"Forced decoder checkpoint path to: {config.model.params.decoder_config.params.ckpt_path}")
        except Exception as e:
            logger.error(f"Error loading or patching configs: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": f"Error loading or patching configs: {str(e)}"}
        
        # Patch the StrLabelConverter class to handle relative paths
        try:
            # Monkey patch the StrLabelConverter.__init__ method to handle relative paths
            from model.utils import StrLabelConverter
            original_init = StrLabelConverter.__init__
            
            def patched_init(self, alphabet, max_text_len, start_id):
                # If alphabet is a relative path, make it absolute
                if isinstance(alphabet, str) and not alphabet.startswith('/'):
                    alphabet = os.path.join(base_repo_dir, alphabet)
                    logger.info(f"Using absolute alphabet path: {alphabet}")
                
                # Call the original init with the fixed path
                original_init(self, alphabet, max_text_len, start_id)
            
            # Apply the monkey patch
            StrLabelConverter.__init__ = patched_init
            logger.info("Patched StrLabelConverter.__init__ to handle relative paths")
        except Exception as e:
            logger.warning(f"Could not patch StrLabelConverter: {str(e)}")
        
        # Patch the VQModel_Decoder class to handle relative paths
        try:
            # Monkey patch the VQModel_Decoder.init_from_ckpt method to handle relative paths
            from model.decoder import VQModel_Decoder
            original_init_from_ckpt = VQModel_Decoder.init_from_ckpt
            
            def patched_init_from_ckpt(self, path):
                # If path is relative, make it absolute
                if not path.startswith('/'):
                    abs_path = os.path.join(base_repo_dir, path)
                    logger.info(f"Converting path in init_from_ckpt: {path} -> {abs_path}")
                    path = abs_path
                
                # Verify the file exists
                if not os.path.exists(path):
                    logger.warning(f"Checkpoint not found at {path}, using main checkpoint as fallback")
                    # Use the main checkpoint as fallback
                    path = "/checkpoints/rsste-finetune.ckpt"
                
                # Call the original method with the fixed path
                original_init_from_ckpt(self, path)
            
            # Apply the monkey patch
            VQModel_Decoder.init_from_ckpt = patched_init_from_ckpt
            logger.info("Patched VQModel_Decoder.init_from_ckpt to handle relative paths")
        except Exception as e:
            logger.warning(f"Could not patch VQModel_Decoder: {str(e)}")
        
        # Initialize model
        try:
            model = instantiate_from_config(config.model).to('cuda')
            model.eval()
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": f"Error initializing model: {str(e)}"}
        
        # Create a simple dataset for inference
        try:
            # Patch the InferenceDataset class to handle relative paths
            from data.dataset import InferenceDataset
            
            # Create a temporary annotation file with the input image and text prompt
            temp_annotation_file = os.path.join(inputs_dir, "temp_annotation.txt")
            with open(temp_annotation_file, "w") as f:
                f.write(f"{os.path.basename(image_path)}\t{text_prompt}")
            
            # Create dataset and dataloader
            dataset = InferenceDataset(config.data.validation.params.size, temp_annotation_file)
            from torch.utils.data import DataLoader
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        except Exception as e:
            logger.error(f"Error creating dataset: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": f"Error creating dataset: {str(e)}"}
        
        # Run inference
        try:
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
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": f"Error during inference: {str(e)}"}

        # Verify the output file exists
        if not os.path.exists(output_path):
            logger.error(f"Output file does not exist: {output_path}")
            return {"error": "Failed to generate output image"}
        
        # Read the output file
        with open(output_path, "rb") as f:
            output_data = f.read()
        
        # Encode the output as base64
        encoded_output = base64.b64encode(output_data).decode("utf-8")
        
        # Clean up the unique repository directory
        try:
            logger.info(f"Cleaning up unique repository directory: {unique_repo_dir}")
            shutil.rmtree(unique_repo_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Error cleaning up unique repository directory: {e}")
        
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
