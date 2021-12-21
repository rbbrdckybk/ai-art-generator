# ai-art-generator
For automating the generation of large batches of AI artwork utilizing VQGAN+CLIP.

# Setup

These instructions were tested on a Windows 10 system with an Nvidia 3080 Ti GPU (12GB VRAM) and 32GB of system memory, but should work on linux systems as well with some minor edits. 

**[1]** Install [Anaconda](https://www.anaconda.com/products/individual), open the root terminal, and create a new environment (and activate it):
```
conda create --name ai-art python=3.9
conda activate ai-art
```

**[2]** Install Pytorch:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
Note that you can customize your Pytorch installation by using [the online tool located here](https://pytorch.org/get-started/locally/).

**[3]** Install other required Python packages:
```
conda install m2-base
conda install -c anaconda git
conda install -c anaconda urllib3
pip install ftfy regex tqdm omegaconf pytorch-lightning IPython kornia imageio imageio-ffmpeg einops torch_optimizer
```

**[4]** Clone this repository and switch to its directory:
```
git clone https://github.com/rbbrdckybk/ai-art-generator
cd ai-art-generator
```

**[5]** Download the default VQGAN pre-trained model checkpoint files:
```
mkdir checkpoints
curl -L -o checkpoints/vqgan_imagenet_f16_16384.yaml -C - "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1"
curl -L -o checkpoints/vqgan_imagenet_f16_16384.ckpt -C - "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1"
```
Note that linux users should replace the double quotes in the curl commands with single quotes.

**[6]** (Optional) Download additional pre-trained models:  
Additional models are not necessary, but provide you with more options. [Here is a good list of available pre-trained models](https://github.com/CompVis/taming-transformers#overview-of-pretrained-models).  
For example, if you also wanted the FFHQ model (trained on faces): 
```
curl -L -o checkpoints/ffhq.yaml -C - "https://app.koofr.net/content/links/0fc005bf-3dca-4079-9d40-cdf38d42cd7a/files/get/2021-04-23T18-19-01-project.yaml?path=%2F2021-04-23T18-19-01_ffhq_transformer%2Fconfigs%2F2021-04-23T18-19-01-project.yaml&force"
curl -L -o checkpoints/ffhq.ckpt -C - "https://app.koofr.net/content/links/0fc005bf-3dca-4079-9d40-cdf38d42cd7a/files/get/last.ckpt?path=%2F2021-04-23T18-19-01_ffhq_transformer%2Fcheckpoints%2Flast.ckpt"
```
You'll also need to install the transformers package for most of these to work:
```
pip install transformers
```
# Usage
