# Changelog

Starting this ~8 months after I created the repo, but I'll do my best to keep this updated with major changes going forward!


## [2022.09.28]
### Fixed
- Forced ESRGAN upscaling to use the same GPU device that was used to create the original artwork to avoid an issue where multiple processes were trying to use the same GPU.
- Fixed a bug that would cause problems if the [style] section of a standard prompt file was left empty or ommitted.

## [2022.08.29]
### Added
- Integrated optional [ESRGAN/GFPGAN](https://github.com/xinntao/Real-ESRGAN) upscaling for use with Stable Diffusion (see [docs](https://github.com/rbbrdckybk/ai-art-generator#advanced-usage)).
- Added a random prompt builder (**random_art.py**) as an alternative method to generate images. See [example](https://github.com/rbbrdckybk/ai-art-generator/blob/main/prompts/random-example.txt) usage until I have a chance to document more fully. Basically you define prompt structure, along with fragments that are randomized and the process will churn out random images until you stop it.

## [2022.08.27]

For this update, I've created my own [/stable-diffusion fork](https://github.com/rbbrdckybk/stable-diffusion) (building on basujindal's excellent low-memory fork). This project now relies on it; **if you have an existing install and want to upgrade, you'll need to delete the /stable-diffusion folder from your main project directory and then [follow the setup instructions starting at step #13](https://github.com/rbbrdckybk/ai-art-generator#setup)**.

### Added
- Added support for a low GPU memory -optimized fork of Stable Diffusion. It's now possible to create 512x512 images with only 4GB of VRAM! (see [docs](https://github.com/rbbrdckybk/ai-art-generator#usage)).
- Added support for Stable Diffusion CUDA device specification.
### Changed
- Removed the NSFW filter from Stable Diffusion.
- Removed the automatic application of invisible image watermarks from Stable Diffusion (and removed invisible-watermark as a setup dependency).
- Hid some warning spam that was enabled by default in SD.
### Fixed
- Fixed the --seed value embedded into image metadata with the actual correct seed used when creating multiple images in batches with SD.

## [2022.08.24]
### Added
- Added support for Stable Diffusion's *--n_iter* parameter (see [docs](https://github.com/rbbrdckybk/ai-art-generator#usage)).
### Changed
- Changed the way that images are saved to subdirectories over multi-day run times. Previously, all images would be saved to your output directory in a subdir date-stamped with the date execution started. Now, output subdirs will always be datestamped relative to actual image creation time. Useful if you're running huge work queues that run for multiple days and you don't want thousands of images in a single directory!

## [2022.08.22]
### Added
- Full support for [Stable Diffusion](https://github.com/CompVis/stable-diffusion) is in (both txt2img and img2img)! Tested and working fine on both Windows 10 and Linux. There are probably minor issues here and there; feel free to open issues as you find them!  

If you have an old installation that you want to upgrade to enable SD support, start at step 13 in the Setup section. You'll likely additionally need to update your transformers package (pip install transformers -U) at minimum, and it doesn't hurt to refresh git repos as well.

## [2022.08.21]
### Added
- Added groundwork for [Stable Diffusion](https://github.com/CompVis/stable-diffusion) support! Public release of the model is tentatively slated for tomorrow, so I'll try to make sure everything works as soon as it's available.
### Fixed
- Fixed a bug introduced with multi-GPU support when running the VQGAN process that would cause a crash on startup.

## [2022.08.20]
### Added
- Added support to specify CUDA device, useful for people with multiple GPUs.
