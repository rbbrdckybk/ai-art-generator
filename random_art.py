# Copyright 2021 - 2022, Bill Kennedy (https://github.com/rbbrdckybk/ai-art-generator)
# SPDX-License-Identifier: MIT

# see random-example.txt in the prompts directory for usage example

import os
import random
import unicodedata
import re
import threading
import time
import shlex
import subprocess
import sys
from datetime import datetime as dt
from datetime import date
from os.path import exists
from pathlib import Path
from PIL.PngImagePlugin import PngImageFile, PngInfo
from torch.cuda import get_device_name

cwd = os.getcwd()
gpu_name = get_device_name()

if sys.platform == "win32" or os.name == 'nt':
    os.environ['PYTHONPATH'] = os.pathsep + (cwd + "\latent-diffusion") + os.pathsep + (cwd + "\\taming-transformers") + os.pathsep + (cwd + "\CLIP")
else:
    os.environ['PYTHONPATH'] = os.pathsep + (cwd + "/latent-diffusion") + os.pathsep + (cwd + "/taming-transformers") + os.pathsep + (cwd + "/CLIP")

SD_LOW_MEMORY = "no"    # use low-memory repo?
SD_LOW_MEM_TURBO = "no" # faster at the cost of ~1GB VRAM (only when SD_LOW_MEMORY = "yes")
WIDTH = 512             # output image width, default is 512
HEIGHT = 512            # output image height, default is 512
STEPS = 80              # number of steps, default = 50
MIN_SCALE = 7.5         # minimum guidance scale, default = 7.5
MAX_SCALE = 7.5         # maximum guidance scale, default = 7.5
SAMPLES = 1             # number of times to sample (e.g.: images to generate)
BATCH_SIZE = 1          # number of images to generate per sample, huge VRAM increase per
MIN_STRENGTH = 0.50     # min strength of starting image influence
MAX_STRENGTH = 0.50     # max strength of starting image influence
DELIM = " "             # default prompt delimiter
USE_UPSCALE = "no"      # upscale output images via ESRGAN/GFPGAN?
UPSCALE_AMOUNT = 2.0    # amount to upscale, default is 2.0x
UPSCALE_FACE_ENH = "no" # use GFPGAN, optimized for faces
UPSCALE_KEEP_ORG = "no" # save the original non-upscaled image when using upscaling?

# directory to write finished output image files
# subdirectories will be automatically created under this folder organized by date
OUTDIR = "output"

# Prevent threads from printing at same time.
print_lock = threading.Lock()

# globals for worker thread control
worker_idle = True
work_done = False
jobs_done = 0

# worker thread executes specified shell command
class Worker(threading.Thread):
    def __init__(self, command, callback=lambda: None):
        threading.Thread.__init__(self)
        self.command = command
        self.callback = callback

    def run(self):
        with print_lock:
            print("Invoking Stable Diffusion, command: " + self.command)

        # create output folder if it doesn't exist
        fullfilepath = self.command.split(" --outdir ",1)[1]
        filepath = fullfilepath.replace(fullfilepath[fullfilepath.rindex('/'):], "")
        Path(filepath).mkdir(parents=True, exist_ok=True)

        do_upscale = False
        face_enh = False
        upscale_keep_orig = False
        if USE_UPSCALE.lower() == "yes":
            do_upscale = True
            if UPSCALE_FACE_ENH.lower() == "yes":
                face_enh = True
            if UPSCALE_KEEP_ORG.lower() == "yes":
                upscale_keep_orig = True

        # invoke Stable Diffusion
        if sys.platform == "win32" or os.name == 'nt':
            subprocess.call(shlex.split(self.command), cwd=(cwd + '\stable-diffusion'))
        else:
            subprocess.call(shlex.split(self.command), cwd=(cwd + '/stable-diffusion'))

        fullfilepath = fullfilepath.replace("../","")
        if do_upscale:
            new_files = os.listdir(fullfilepath + "/gpu_0")
            if len(new_files) > 0:
                upscale(UPSCALE_AMOUNT, fullfilepath + "/gpu_0", face_enh)

                # remove originals if upscaled version present
                new_files = os.listdir(fullfilepath + "/gpu_0")
                for f in new_files:
                    if (".png" in f):
                        basef = f.replace(".png", "")
                        if basef[-2:] == "_u":
                            # this is an upscaled image, delete the original
                            # or save it in /original if desired
                            if exists(fullfilepath + "/gpu_0/" + basef[:-2] + ".png"):
                                if upscale_keep_orig:
                                    # move the original to /original
                                    orig_dir = fullfilepath + "/original"
                                    Path(orig_dir).mkdir(parents=True, exist_ok=True)
                                    os.replace(fullfilepath + "/gpu_0/" + basef[:-2] + ".png", \
                                        orig_dir + "/" + basef[:-2] + ".png")
                                else:
                                    os.remove(fullfilepath + "/gpu_0/" + basef[:-2] + ".png")

        # find the new image(s) that SD created: re-name, process, and move them
        new_files = os.listdir(fullfilepath + "/gpu_0")
        nf_count = 0

        # save just the essential prompt params to metadata
        meta_prompt = self.command.split(" --prompt ",1)[1]
        meta_prompt = meta_prompt.split(" --outdir ",1)[0]

        upscale_text = ""
        if do_upscale:
            upscale_text = " (upscaled "
            upscale_text += str(UPSCALE_AMOUNT) + "x via "
            if face_enh:
                upscale_text += "GFPGAN)"
            else:
                upscale_text += "ESRGAN)"

        for f in new_files:
            if (".png" in f):
                pngImage = PngImageFile(fullfilepath + "/gpu_0/" + f)
                im = pngImage.convert('RGB')
                exif = im.getexif()
                exif[0x9286] = meta_prompt
                exif[0x9c9c] = meta_prompt.encode('utf16')
                exif[0x9c9d] = ('AI art (' + gpu_name + ')' + upscale_text).encode('utf16')
                exif[0x0131] = "https://github.com/rbbrdckybk/ai-art-generator"
                newfilename = dt.now().strftime('%Y%m-%d%H-%M%S-') + str(nf_count)
                nf_count += 1
                im.save(fullfilepath + "/" + newfilename + ".jpg", exif=exif, quality=88)
                if exists(fullfilepath + "/gpu_0/" + f):
                    os.remove(fullfilepath + "/gpu_0/" + f)
        try:
            os.rmdir(fullfilepath + "/gpu_0")
        except OSError as e:
            # nothing to do here, we only want to remove the dir if empty
            pass

        with print_lock:
            print("Worker done.")
        self.callback()


# ESRGAN/GFPGAN upscaling:
# scale - upscale by this amount, default is 2.0x
# dir - upscale all images in this folder
# do_face_enhance - True/False use GFPGAN (for faces)
def upscale(scale, dir, do_face_enhance):
    command = "python inference_realesrgan.py -n RealESRGAN_x4plus --suffix u -s "

    # check that scale is a valid float, otherwise use default scale of 4
    try :
        float(scale)
        command += str(scale)
    except :
        command += "2"

    # append the input/output dir
    command += " -i ..//" + dir + " -o ..//" + dir

    # whether to use GFPGAN for faces
    if do_face_enhance:
        command += " --face_enhance"

    cwd = os.getcwd()
    print ("Invoking Real-ESRGAN: " + command)

    # invoke Real-ESRGAN
    if sys.platform == "win32" or os.name == 'nt':
        subprocess.call(shlex.split(command), cwd=(cwd + '\Real-ESRGAN'), stderr=subprocess.DEVNULL)
    else:
        subprocess.call(shlex.split(command), cwd=(cwd + '/Real-ESRGAN'), stderr=subprocess.DEVNULL)


# maintains the info in each input file [prompt] section
class PromptSection():
    def __init__(self, tokens, min_pick, max_pick, delim):
        self.tokens = tokens
        self.min_pick = min_pick
        self.max_pick = max_pick
        self.delim = delim

    def debug_print(self):
        print("\n*** Printing contents of PromptSection: ***")
        print("min pick: " + str(self.min_pick))
        print("max pick: " + str(self.max_pick))
        print("delim: (" + self.delim + ')')
        if len(self.tokens) > 0:
            print("tokens:")
            for x in self.tokens:
                print(">> " + x)
        else:
            print("tokens list is empty")
        print('\n')

# for easy management of input files
# input_path is the directory of the input images to use
class InputManager():
    def __init__(self, input_path):
        # a list of all the files we're using as inputs
        self.files = list()
        if input_path != "":
            self.input_directory = input_path

            # populate the list with the given init directory
            for x in os.listdir(input_path):
                if x.endswith('.jpg') or x.endswith('.png'):
                    self.files.append(x)

    # pick a random file from the list
    def pick_random(self):
        if len(self.files) > 0:
            x = random.randint(0, len(self.files)-1)
            return self.files[x]
        else:
            return ""

    def debug_print_files(self):
        if len(self.files) > 0:
            print("Listing " + str(len(self.files)) + " total files in '" + self.input_directory + "':")
            for x in self.files:
                print(x)
        else:
            print("Input image directory '" + self.input_directory + "' is empty; input images will not be used.")

# for easy management of prompts
class PromptManager():
    def __init__(self, prompt_file):
        # text file containing all of the prompt/style/etc info
        self.prompt_file_name = prompt_file

        # list for config info
        self.config = list()
        # list of PromptSection
        self.prompts = list()

        self.__init_config(self.config, "config")
        self.__init_prompts(self.prompts, "prompts")

        #self.debug_print()

    # init the prompts list
    def __init_prompts(self, which_list, search_text):
        with open(self.prompt_file_name) as f:
            lines = f.readlines()

            found_header = False
            search_header = '[' + search_text

            tokens = list()
            ps = PromptSection(tokens, 1, 1, DELIM)

            # find the search text and read until the next search header
            for line in lines:
                # ignore comments and strip whitespace
                line = line.strip().split('#', 1)
                line = line[0]

                # if we already found the header we want and we see another header,
                if found_header and len(line) > 0 and line.startswith('['):
                    # save PromptSection
                    which_list.append(ps)

                    # start a new PS
                    tokens = list()
                    ps = PromptSection(tokens, 1, 1, DELIM)

                    # look for next prompt section
                    found_header = False

                # found the search header
                if search_header.lower() in line.lower() and line.endswith(']'):
                    found_header = True

                    # check for optional args
                    args = line.strip(search_header).strip(']').strip()
                    vals = shlex.split(args, posix=False)

                    # grab min/max args
                    if len(vals) > 0:
                        if '-' in vals[0]:
                            minmax = vals[0].split('-')
                            if len(minmax) > 0:
                                ps.min_pick = minmax[0].strip()
                                if len(minmax) > 1:
                                    ps.max_pick = minmax[1].strip()
                        else:
                            ps.min_pick = vals[0]
                            ps.max_pick = vals[0]

                        # grab delim arg
                        if len(vals) > 1:
                            if vals[1].startswith('\"') and vals[1].endswith('\"'):
                                ps.delim = vals[1].strip('\"')

                    line = ""

                if len(line) > 0 and found_header:
                    ps.tokens.append(line)

            # save final PromptSection if not empty
            if len(ps.tokens) > 0:
                which_list.append(ps)


    # init the config list
    def __init_config(self, which_list, search_text):
        with open(self.prompt_file_name) as f:
            lines = f.readlines()

            search_header = '[' + search_text + ']'
            found_header = False

            # find the search text and read until the next search header
            for line in lines:
                # ignore comments and strip whitespace
                line = line.strip().split('#', 1)
                line = line[0]

                # if we already found the header we want and we see another header, stop
                if found_header and len(line) > 0 and line[0] == '[':
                    break

                # found the search header
                if search_header.lower() == line.lower():
                    found_header = True
                    line = ""

                if len(line) > 0 and found_header:
                    #print(search_header + ": " + line)
                    which_list.append(line)

    def debug_print(self):
        if len(self.prompts) > 0:
            print("\nPS contents:\n")
            for x in self.prompts:
                x.debug_print()
        else:
            print("prompts list is empty")

    # update config variables if there were changes in the prompt file
    def handle_config(self):
        if len(self.config) > 0:
            for line in self.config:
                # ignore comments and strip whitespace
                line = line.strip().split('#', 1)
                line = line[0]

                # update config values for found directives
                if '=' in line:
                    line = line.split('=', 1)
                    command = line[0].lower().strip()
                    value = line[1].strip()

                    if command == 'width':
                        if value != '':
                            global WIDTH
                            WIDTH = value

                    elif command == 'height':
                        if value != '':
                            global HEIGHT
                            HEIGHT = value

                    elif command == 'steps':
                        if value != '':
                            global STEPS
                            STEPS = value

                    elif command == 'min_scale':
                        if value != '':
                            global MIN_SCALE
                            MIN_SCALE = value

                    elif command == 'max_scale':
                        if value != '':
                            global MAX_SCALE
                            MAX_SCALE = value

                    elif command == 'samples':
                        if value != '':
                            global SAMPLES
                            SAMPLES = value

                    elif command == 'batch_size':
                        if value != '':
                            global BATCH_SIZE
                            BATCH_SIZE = value

                    elif command == 'min_strength':
                        if value != '':
                            global MIN_STRENGTH
                            MIN_STRENGTH = value

                    elif command == 'max_strength':
                        if value != '':
                            global MAX_STRENGTH
                            MAX_STRENGTH = value

                    elif command == 'sd_low_memory':
                        if value != '':
                            global SD_LOW_MEMORY
                            SD_LOW_MEMORY = value

                    elif command == 'sd_low_mem_turbo':
                        if value != '':
                            global SD_LOW_MEM_TURBO
                            SD_LOW_MEM_TURBO = value

                    elif command == 'use_upscale':
                        if value != '':
                            global USE_UPSCALE
                            USE_UPSCALE = value

                    elif command == 'upscale_amount':
                        if value != '':
                            global UPSCALE_AMOUNT
                            UPSCALE_AMOUNT = value

                    elif command == 'upscale_face_enh':
                        if value != '':
                            global UPSCALE_FACE_ENH
                            UPSCALE_FACE_ENH = value

                    elif command == 'upscale_keep_org':
                        if value != '':
                            global UPSCALE_KEEP_ORG
                            UPSCALE_KEEP_ORG = value

                    elif command == 'delim':
                        if value != '':
                            if value.startswith('\"') and value.endswith('\"'):
                                global DELIM
                                DELIM = value.strip('\"')
                                #print("New delim: \"" + DELIM + "\"")
                            else:
                                print("\n*** WARNING: prompt file command DELIM value (" + value + ") not understood (make sure to put quotes around it)! ***\n")
                                time.sleep(1.5)

                    else:
                        print("\n*** WARNING: prompt file command not recognized: " + command.upper() + " (it will be ignored)! ***\n")
                        time.sleep(1.5)


    # create a random prompt from the information in the prompt file
    def pick_random(self):
        fragments = 0
        full_prompt = ""
        tokens = list()

        if len(self.prompts) > 0:
            # iterate through each PromptSection to build the prompt
            for ps in self.prompts:
                fragment = ""
                picked = 0
                # decide how many tokens to pick
                x = random.randint(int(ps.min_pick), int(ps.max_pick))

                # pick token(s)
                if len(ps.tokens) >= x:
                    tokens = ps.tokens.copy()
                    for i in range(x):
                        z = random.randint(0, len(tokens)-1)
                        if picked > 0:
                            fragment += ps.delim
                        fragment += tokens[z]
                        del tokens[z]
                        picked += 1
                else:
                    # not enough tokens to take requested amount, take all
                    for t in ps.tokens:
                        if picked > 0:
                            fragment += ps.delim
                        fragment += t
                        picked += 1

                # add this fragment to the overall prompt
                if fragment != "":
                    if fragments > 0:
                        if not (fragment.startswith(',') or fragment.startswith(';')):
                            full_prompt += DELIM
                    full_prompt += fragment
                    fragments += 1

        full_prompt = full_prompt.replace(",,", ",")
        full_prompt = full_prompt.replace(", ,", ",")
        full_prompt = full_prompt.replace(" and,", ",")
        full_prompt = full_prompt.replace(" by and ", " by ")
        full_prompt = full_prompt.strip().strip(',')

        return full_prompt

# creates the full command to invoke SD with our specified params
# and selected prompt + input image
def create_command(prompt, input_image, output_dir_ext):
    seed = random.randint(1, 2**32) - 1000
    output_folder = OUTDIR + '/' + str(date.today()) + '-' + str(output_dir_ext)
    basefilepath = output_folder + '/' + slugify(prompt)
    fullfilepath = basefilepath + ".png"
    x = 0

    # check to see if output file already exists; find unique name if it does
    while exists(fullfilepath):
        x += 1
        fullfilepath = basefilepath + '-' + str(x) + ".png"

    base_command = "python scripts_mod/txt2img.py"
    if str(SD_LOW_MEMORY).lower() == "yes":
        base_command = "python scripts_mod/optimized_txt2img.py"

    if input_image != "":
        base_command = "python scripts_mod/img2img.py"
        if str(SD_LOW_MEMORY).lower() == "yes":
            base_command = "python scripts_mod/optimized_img2img.py"

    set_scale = round(random.uniform(float(MIN_SCALE), float(MAX_SCALE)), 1)
    set_strength = round(random.uniform(float(MIN_STRENGTH), float(MAX_STRENGTH)), 2)

    if SD_LOW_MEMORY.lower() == "yes" and SD_LOW_MEM_TURBO.lower() == "yes":
        base_command += " --turbo"

    command = base_command + " --skip_grid" \
        + " --n_iter " + str(SAMPLES) \
        + " --n_samples " + str(BATCH_SIZE) \
        + " --prompt \"" + prompt + "\"" \
        + " --ddim_steps " + str(STEPS) \
        + " --scale " + str(set_scale) \
        + " --seed " + str(seed)

    if input_image != "":
        command += " --init-img \"../" + input_image + "\"" + " --strength " + str(set_strength)
    else:
        command += " --W " + str(WIDTH) + " --H " + str(HEIGHT)

    command += " --outdir ../" + output_folder

    return command

# Taken from https://github.com/django/django/blob/master/django/utils/text.py
# Using here to make filesystem-safe directory names
def slugify(value, allow_unicode=False):
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')

# starts a new worker thread
def do_work(command):
    global worker_idle
    worker_idle = False
    thread = Worker(command, on_work_done)
    thread.start()

# callback for worker threads that finish their job
def on_work_done():
    global worker_idle
    global jobs_done
    worker_idle = True
    jobs_done += 1

# entry point
if __name__ == '__main__':
    if len(sys.argv) > 1:
        prompt_filename = sys.argv[1]

        input_folder = ""
        if len(sys.argv) > 2:
            input_folder = sys.argv[2]

        if not exists(prompt_filename):
            print("\nThe specified prompt file '" + prompt_filename + "' doesn't exist!")
            print("Please specify a valid text file containing your prompt information.")
            exit()

        if input_folder != "":
            if not exists(input_folder):
                print("\nThe specified input folder '" + input_folder + "' doesn't exist!")
                print("Please specify a valid directory containing your input images.")
                exit()

        prompts = PromptManager(prompt_filename)
        prompts.handle_config()
        inputs = InputManager(input_folder)
        output_ext = prompt_filename.strip().split('.', 1)[0]

        # start work loop
        while not work_done:
            if worker_idle:
                # worker is idle, start some work
                print("\nStarting job #" + str(jobs_done+1) + ":")
                fullinputpath = ""
                if input_folder != "":
                    fullinputpath = input_folder + '/' + inputs.pick_random()

                # if prompt file is in subdir, remove all but final
                # subdir name for output folding naming purposes
                if '/' in output_ext:
                    output_ext = output_ext.strip().rsplit('/', 1)
                    output_ext = output_ext[1]
                if '\\' in output_ext:
                    output_ext = output_ext.strip().rsplit('\\', 1)
                    output_ext = output_ext[1]

                new_command = create_command(prompts.pick_random(), fullinputpath, output_ext)
                worker_idle = False
                do_work(new_command)
            else:
                # wait for worker to finish
                time.sleep(.025)

        # this will currently never happen is there is no condition that would
        # ever set work_done to true - use cntrl+c to kill
        print('\nAll work done!')

    else:
        print("\nUsage: python random_art.py [prompt-file] [path-to-input-image-folder (optional)]")
        print("Example: python random_art.py prompts.txt inputs/faces")

    exit()
