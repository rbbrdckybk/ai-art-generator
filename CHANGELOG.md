# Changelog

I'll do my best to keep this updated with major changes.


## [0.1.1] - 2022-08-21
### Added
- Added groundwork for [Stable Diffusion](https://github.com/CompVis/stable-diffusion) support! Public release of the model is tentatively slated for tomorrow, so I'll try to make sure everything works as soon as it's available.
### Fixed
- Fixed a bug introduced with multi-GPU support when running the VQGAN process that would cause a crash on startup.

## [0.1.0] - 2022-08-20
### Added
- Added support to specify CUDA device, useful for people with multiple GPUs.
