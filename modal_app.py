import os
import modal
import base64
from fastapi import Request
#he
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
    "transformers",       # Common dependency for transformer models
    "pytest"
)
#KEEP THE PAIN FRESH
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
        # Ensure checkpoint is downloaded
        if not os.path.exists("/checkpoints/rsste-finetune.ckpt"):
            logger.info("Downloading checkpoint...")
            checkpoint_url = "https://huggingface.co/v4mmko/RS-STE/resolve/main/rsste-finetune.ckpt"
            subprocess.run(f"wget {checkpoint_url} -O /checkpoints/rsste-finetune.ckpt", shell=True, check=True)
            logger.info("Checkpoint downloaded successfully")

        # Ensure repository is downloaded and patched
        repo_dir = "/checkpoints/RS-STE"
        if not os.path.exists(repo_dir) or not os.path.exists(f"{repo_dir}/model/utils.py"):
            logger.info("Downloading and preparing repository...")
            # Remove existing repo if it exists but is incomplete
            if os.path.exists(repo_dir):
                subprocess.run(f"rm -rf {repo_dir}", shell=True, check=True)
            
            # Clone the repository
            subprocess.run(
                f"git clone https://github.com/xunmengshe2x/rs-ste {repo_dir}",
                shell=True,
                check=True
            )
            
            # Verify the repository was cloned successfully
            if not os.path.exists(repo_dir):
                raise FileNotFoundError("Repository directory was not created after git clone")
            
            # List contents to verify
            logger.info(f"Repository contents: {os.listdir(repo_dir)}")
            
            # Verify model directory exists
            if not os.path.exists(f"{repo_dir}/model"):
                raise FileNotFoundError("Model directory not found in cloned repository")
            
            # Verify utils.py exists
            utils_path = f"{repo_dir}/model/utils.py"
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
                    main_path = "/checkpoints/RS-STE/main.py"
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
            decoder_config_path = "/checkpoints/RS-STE/configs/vqgan_decoder.yaml"
            synth_pair_path = "/checkpoints/RS-STE/configs/synth_pair.yaml"
            
            if not os.path.exists(decoder_config_path):
                logger.error(f"Decoder config not found at {decoder_config_path}")
                return {"error": f"Decoder config not found at {decoder_config_path}"}
            
            if not os.path.exists(synth_pair_path):
                logger.error(f"Synth pair config not found at {synth_pair_path}")
                return {"error": f"Synth pair config not found at {synth_pair_path}"}
            
            # Load and patch configs with absolute paths
            decoder_config = OmegaConf.load(decoder_config_path)
            config = OmegaConf.load(synth_pair_path)
            
            # Patch relative paths in configs to absolute paths
            def fix_paths_in_config(cfg, base_path):
                """Recursively fix relative paths in config to absolute paths"""
                if isinstance(cfg, dict):
                    for key, value in cfg.items():
                        if isinstance(value, (dict, list)):
                            fix_paths_in_config(value, base_path)
                        elif isinstance(value, str) and (
                            "data/" in value or 
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
            fix_paths_in_config(decoder_config, "/checkpoints/RS-STE")
            fix_paths_in_config(config, "/checkpoints/RS-STE")
            
            # Specifically check and fix the alphabet path
            if hasattr(config.model.params, 'alphabet') and not config.model.params.alphabet.startswith('/'):
                alphabet_path = config.model.params.alphabet
                abs_alphabet_path = os.path.join("/checkpoints/RS-STE", alphabet_path)
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
            
            # Set decoder config and checkpoint path
            config.model.params.decoder_config = decoder_config.model
            config.model.params.ckpt_path = "/checkpoints/rsste-finetune.ckpt"
            
            # Log the patched configs
            logger.info(f"Patched config: {OmegaConf.to_yaml(config)}")
        except Exception as e:
            logger.error(f"Error loading or patching configs: {str(e)}")
            return {"error": f"Error loading or patching configs: {str(e)}"}
        
        # Patch the StrLabelConverter class to handle relative paths
        try:
            # Monkey patch the StrLabelConverter.__init__ method to handle relative paths
            from model.utils import StrLabelConverter
            original_init = StrLabelConverter.__init__
            
            def patched_init(self, alphabet, max_text_len, start_id):
                # If alphabet is a relative path, make it absolute
                if isinstance(alphabet, str) and not alphabet.startswith('/'):
                    alphabet = os.path.join("/checkpoints/RS-STE", alphabet)
                    logger.info(f"Using absolute alphabet path: {alphabet}")
                
                # Call the original init with the fixed path
                original_init(self, alphabet, max_text_len, start_id)
            
            # Apply the monkey patch
            StrLabelConverter.__init__ = patched_init
            logger.info("Patched StrLabelConverter.__init__ to handle relative paths")
        except Exception as e:
            logger.warning(f"Could not patch StrLabelConverter: {str(e)}")
        
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
