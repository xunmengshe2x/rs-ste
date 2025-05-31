# Recognition-Synergistic Scene Text Editing 🎨✨
[![arXiv](https://img.shields.io/badge/arXiv-2503.08387-b31b1b.svg)](https://arxiv.org/abs/2503.08387)

**CVPR 2025 | ​Official Implementation**
This is an official implementation of RS-STE proposed by our paper Recognition-Synergistic Scene Text Editing (CVPR 2025).

<div style="text-align: center;">
    <img src="docs/examples.png" alt="examples" style="width: 67%;" />
</div>



---
## 🚧 Under Construction 🚧
We're currently polishing the code and preparing everything for public release. Here's what you can expect in the near future:
**TODO List ✅**
- [x] Inference demo.
- [ ] Training code and dataset preparation.
- [ ] Evaluation scripts.

Stay tuned! We'll update this repository step by step. ⏳

---

## 0️⃣ Install

**Environment🌄**
You can use uv to create the virtual environment:
```bash
pip install uv # install uv
uv venv # create virtual environment
source .venv/bin/activate
# Our CUDA version is 11.4
uv pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
uv pip install -r requirements.txt
```

Or use conda:
```bash
conda create -n rsste python=3.8
conda activate rsste
# Our CUDA version is 11.4
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

## 1️⃣ Dataset 
Organize the annotation file into the following format and save it directly to ```data/annotation```. An example annotation file for inference is given as:
```
import pickle
data = {
    "image1_paths":['example_data/3.png', 'example_data/47.png', 'example_data/81.png', 'example_data/91.png', 'example_data/106.png', 'example_data/133.png'], # source images
    "image2_paths":[], # (Optional, needed during pretraining stage with paired images.) # target images
    "image1_rec":['MAIL', 'Colchester', 'Council', 'Analysis', 'LEISURE', 'RECIPES'], # (Optional, needed during all the training stage) # source image texts
    "image2_rec":['ROYAL', 'Insurance', 'County', 'Mining', 'LIMITED', 'FESTIVE'] # target text
}
with open("data/annotation/inference_annotations.pkl", "wb") as f:
    pickle.dump(data, f)
```
you can run ```uv run python tools/dataset_prepare.py``` to get the annotation file of examples.


## 2️⃣ Inference
Download the pretrained checkpoints here: [Pretrained Model Download](https://pan.baidu.com/s/151EXQY5SdpETd3BS62dJYQ?pwd=db8s).
Then run:
```
uv run python inference.py --resume path/to/checkpoint
```

## 3️⃣ Training

## 🎬 Citation

