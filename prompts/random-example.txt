# *******************************************************************************
# random prompt generator example
# this will generate images using Stable Diffusion forever until stopped
# the prompts will be generated randomly based on the information defined below
# *******************************************************************************

# run with this command from the main ai-art-generator dir:
#   python random_art.py prompts/random-example.txt

# alternatively, you can also specify a folder of input images to use:
#   python random_art.py prompts/random-example.txt [path to input images folder]
# all images should be the same height/width
# an input image will be chosen at random for each prompt

# *******************************************************************************
# config goes here
# this is optional; will use defaults if not provided
# *******************************************************************************
[config]

SD_LOW_MEMORY = no	# use low-memory repo? much slower but requires less VRAM
WIDTH = 384             # output image width, default is 512
HEIGHT = 384            # output image height, default is 512
STEPS = 50              # number of steps, default = 50
MIN_SCALE = 7           # minimum guidance scale, default = 7.5
MAX_SCALE = 12		# maximum guidance scale, default = 7.5
SAMPLES = 3		# number of images to generate per prompt
DELIM = " "		# delimiter to use between prompt sections, default is space

# these only apply if an input image is specified
MIN_STRENGTH = 0.55     # min strength of starting image influence
MAX_STRENGTH = 0.75     # max strength of starting image influence

# ESRGAN/GFPGAN upscaling settings
USE_UPSCALE = no	# use upscaling?
UPSCALE_AMOUNT = 2.0	# upscale amount, default = 2.0x
UPSCALE_FACE_ENH = no	# use GFPGAN (best for faces)?
UPSCALE_KEEP_ORG = no	# keep original image when upscaling?


# *******************************************************************************
# example first prompt section
# one of these lines will be chosen at random to build the first part of your prompt
# *******************************************************************************
[prompts]

a man
a woman
a cute puppy

# *******************************************************************************
# example next prompt section
# since there is only one line specified under this [prompts] section, 
# the below text will always appear at this point in your prompt
# *******************************************************************************
[prompts]

, art by

# *******************************************************************************
# example prompt section
# this time with two optional parameters
# 1-2 means between 1 and 2 of these items will be chosen at random
# " and " will be used to join the chosen items together
# *******************************************************************************
[prompts 1-2 " and "]

Artgerm
Greg Rutkowski
WLOP

# *******************************************************************************
# example prompt section
# this comma will always appear here since there is nothing else to choose from
# *******************************************************************************
[prompts]

,

# *******************************************************************************
# another example prompt section
# 0-4 means between 0-4 of these items will be chosen at random
# ", " will be used to join the chosen items together
# *******************************************************************************
[prompts 0-4 ", "]

trending on artstation
surrealism
illustration
digital painting
comic book art
art nouveau
whimsical
pop surrealism
whimsical
volumetric lighting
beautiful rim lighting
photorealistic

# *******************************************************************************
# you can continue to add as many additional [prompts] sections as you want
# given the example prompts above, an example prompt that may be produced is:
#
# "a cute puppy, art by Greg Rutkowski and Artgerm, illustration, whimsical, trending on artstation"
#
# *******************************************************************************
