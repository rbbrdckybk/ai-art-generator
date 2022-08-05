# AI Art Generator
For automating the creation of large batches of AI-generated artwork utilizing VQGAN+CLIP and CLIP-guided diffusion locally.  
Some example images that I've created via this process (these are cherry-picked and sharpened):  
<img src="/samples/sample01.jpg" width="366">
<img src="/samples/sample02.jpg" width="220">
<img src="/samples/sample03.jpg" width="220">
<img src="/samples/sample04.jpg" width="220">
<img src="/samples/sample05.jpg" width="220">
<img src="/samples/sample06.jpg" width="366">

# Requirements

You'll need an Nvidia GPU, preferably with a decent amount of VRAM. 12GB of VRAM is sufficient for 512x512 output images, and 8GB should be enough for 384x384. To generate 1024x1024 images, you'll need ~24GB of VRAM. Generating small images and then upscaling via [ESRGAN](https://github.com/xinntao/Real-ESRGAN) or some other package provides very good results as well.

It should be possible to run on an AMD GPU, but you'll need to be on Linux to install the ROCm version of Pytorch. I don't have an AMD GPU to throw into a Linux machine so I haven't tested this myself.

# Setup

These instructions were tested on a Windows 10 desktop with an Nvidia 3080 Ti GPU (12GB VRAM), and also on an Ubuntu Server 20.04.3 system with an old Nvidia Tesla M40 GPU (24GB VRAM).

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
conda install -c anaconda git urllib3
pip install torchvision transformers keyboard pillow ftfy regex tqdm omegaconf pytorch-lightning IPython kornia imageio imageio-ffmpeg einops torch_optimizer
```

**[4]** Clone this repository and switch to its directory:
```
git clone https://github.com/rbbrdckybk/ai-art-generator
cd ai-art-generator
```
Note that Linux users may need single quotes around the URL in the clone command.

**[5]** Clone additional required repositories:
```
git clone https://github.com/openai/CLIP
git clone https://github.com/CompVis/taming-transformers
```

**[6]** Download the default VQGAN pre-trained model checkpoint files:
```
mkdir checkpoints
curl -L -o checkpoints/vqgan_imagenet_f16_16384.yaml -C - "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1"
curl -L -o checkpoints/vqgan_imagenet_f16_16384.ckpt -C - "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1"
```
Note that Linux users should replace the double quotes in the curl commands with single quotes.

**[7]** (Optional) Download additional pre-trained models:  
Additional models are not necessary, but provide you with more options. [Here is a good list of available pre-trained models](https://github.com/CompVis/taming-transformers#overview-of-pretrained-models).  
For example, if you also wanted the FFHQ model (trained on faces): 
```
curl -L -o checkpoints/ffhq.yaml -C - "https://app.koofr.net/content/links/0fc005bf-3dca-4079-9d40-cdf38d42cd7a/files/get/2021-04-23T18-19-01-project.yaml?path=%2F2021-04-23T18-19-01_ffhq_transformer%2Fconfigs%2F2021-04-23T18-19-01-project.yaml&force"
curl -L -o checkpoints/ffhq.ckpt -C - "https://app.koofr.net/content/links/0fc005bf-3dca-4079-9d40-cdf38d42cd7a/files/get/last.ckpt?path=%2F2021-04-23T18-19-01_ffhq_transformer%2Fcheckpoints%2Flast.ckpt"
```

**[8]** (Optional) Test VQGAN+CLIP:  
```
python vqgan.py -s 128 128 -i 200 -p "a red apple" -o output/output.png
```
You should see output.png created in the output directory, which should loosely resemble an apple.

**[9]** Install packages for CLIP-guided diffusion (if you're only interested in VQGAN+CLIP, you can skip everything from here to the end): 
```
pip install ipywidgets omegaconf torch-fidelity einops wandb opencv-python matplotlib lpips datetime timm
conda install pandas
```

**[10]** Clone repositories for CLIP-guided diffusion:
```
git clone https://github.com/crowsonkb/guided-diffusion
git clone https://github.com/assafshocher/ResizeRight
git clone https://github.com/CompVis/latent-diffusion
```

**[11]** Download models needed for CLIP-guided diffusion:
```
mkdir content\models
curl -L -o content/models/256x256_diffusion_uncond.pt -C - "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
curl -L -o content/models/512x512_diffusion_uncond_finetune_008100.pt -C - "http://batbot.tv/ai/models/guided-diffusion/512x512_diffusion_uncond_finetune_008100.pt"
curl -L -o content/models/secondary_model_imagenet_2.pth -C - "https://ipfs.pollinations.ai/ipfs/bafybeibaawhhk7fhyhvmm7x24zwwkeuocuizbqbcg5nqx64jq42j75rdiy/secondary_model_imagenet_2.pth"
mkdir content\models\superres
curl -L -o content/models/superres/project.yaml -C - "https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1"
curl -L -o content/models/superres/last.ckpt -C - "https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1"
```
Note that Linux users should again replace the double quotes in the curl commands with single quotes.

**[12]** (Optional) Test CLIP-guided diffusion:  
```
python diffusion.py -s 128 128 -i 200 -p "a red apple" -o output.png
```
You should see output.png created in the output directory, which should loosely resemble an apple.

# Usage

Essentially, you just need to create a text file containing the subjects and styles you want to use to generate images. If you have 5 subjects and 20 styles in your prompt file, then a total of 100 output images will be created (20 style images for each subject).

Take a look at **example-prompts.txt** to see how prompt files should look. You can ignore everything except the [subjects] and [styles] areas for now. Lines beginning with a '#' are comments and will be ignored, and lines beginning with a '!' are settings directives and are explained in the next section. For now, just modify the example subjects and styles with whatever you'd like to use.

After you've populated **example-prompts.txt** to your liking, you can simply run:
```
python make_art.py example-prompts.txt
```
Depending on your hardware and settings, each image will take anywhere from a few minutes to a few hours (on older hardware) to create.

Output images are created in the **output/[current date]-[prompt file name]/** directory by default. The output directory will contain a JPG file for each image named for the subject & style used to create it. So for example, if you have "a monkey on a motorcycle" as one of your subjects, and "by Picasso" as a style, the output image will be created as output/[current date]-[prompt file name]/a-monkey-on-a-motorcycle-by-picasso.jpg.

You can press **CTRL+SHIFT+P** any time to pause execution (the pause will take effect when the current image is finished rendering). Press **CTRL+SHIFT+P** again to unpause. Useful if you're running this on your primary computer and need to use your GPU for something else for awhile. You can also press **CTRL+SHIFT+R** to reload the prompt file if you've changed it (the current work queue will be discarded, and a new one will be built from the contents of your prompt file). **Note that keyboard input only works on Windows.**

The settings used to create each image are saved as metadata in each output JPG file by default. You can read the metadata info back by using any EXIF utility, or by simply right-clicking the image file in Windows Explorer and selecting "properties", then clicking the "details" pane. The "comments" field holds the command used to create the image.

# Advanced Usage

Directives can be included in your prompt file to modify settings for all prompts that follow it. These settings directives are specified by putting them on their own line inside of the [subject] area of the prompt file, in the following format:  

**![setting to change] = [new value]**  

For **[setting to change]**, valid directives are:  
 * PROCESS
 * WIDTH
 * HEIGHT
 * ITERATIONS
 * CUTS
 * INPUT_IMAGE
 * LEARNING_RATE (vqgan only)
 * TRANSFORMER (vqgan only)
 * OPTIMISER (vqgan only)
 * CLIP_MODEL (vqgan only)
 * D_VITB16, D_VITB32, D_RN101, D_RN50, D_RN50x4, D_RN50x16 (diffusion only)

Some examples: 
```
!PROCESS = vqgan
```
This will set the current AI image-generation process. Valid options are **vqgan** or **diffusion**.
```
!WIDTH = 384
!HEIGHT = 384
```
This will set the output image size to 384x384. A larger output size requires more GPU VRAM.
```
!TRANSFORMER = ffhq
```
This will tell VQGAN to use the FFHQ transformer (somewhat better at faces), instead of the default (vqgan_imagenet_f16_16384). You can follow step 7 in the setup instructions above to get the ffhq transformer, along with a link to several others.

Whatever you specify here MUST exist in the checkpoints directory as a .ckpt and .yaml file.
```
!INPUT_IMAGE = samples/face-input.jpg
```
This will use samples/face-input.jpg (or whatever image you specify) as the starting image, instead of the default random white noise. Input images must be the same aspect ratio as your output images for good results.
```
!INPUT_IMAGE = 
```
Setting any of these values to nothing will return it to its default. So in this example, no starting image will be used.

TODO: finish settings examples & add usage tips/examples
