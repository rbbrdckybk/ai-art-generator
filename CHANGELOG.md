# Changelog

I'll do my best to keep this updated with major changes.

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
