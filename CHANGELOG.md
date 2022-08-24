# Changelog

Starting this ~8 months after I created the repo, but I'll do my best to keep this updated with major changes going forward!

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
