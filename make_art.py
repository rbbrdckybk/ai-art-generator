# Copyright 2021 - 2022, Bill Kennedy (https://github.com/rbbrdckybk/ai-art-generator)
# SPDX-License-Identifier: MIT

import threading
import time
import datetime
import shlex
import subprocess
import sys
import unicodedata
import re
import random
import os
from os.path import exists
from datetime import datetime as dt
from datetime import date
from pathlib import Path
from collections import deque
from PIL.PngImagePlugin import PngImageFile, PngInfo
from torch.cuda import get_device_name

# for stable diffusion
cwd = os.getcwd()

if sys.platform == "win32" or os.name == 'nt':
    import keyboard
    os.environ['PYTHONPATH'] = os.pathsep + (cwd + "\latent-diffusion") + os.pathsep + (cwd + "\\taming-transformers") + os.pathsep + (cwd + "\CLIP")
else:
    os.environ['PYTHONPATH'] = os.pathsep + (cwd + "/latent-diffusion") + os.pathsep + (cwd + "/taming-transformers") + os.pathsep + (cwd + "/CLIP")

# these can be overriden with prompt file directives, no need to change them here
CUDA_DEVICE = 0         # cuda device to use, default is 0
PROCESS = "vqgan"       # which AI process to use, default is vqgan
WIDTH = 512             # output image width, default is 512
HEIGHT = 512            # output image height, default is 512
ITERATIONS = 500        # number of times to run, default is 500 (VQGAN/DIFFUSION ONLY)
SEED = -1               # initialize with a specific seed value instead of random?
CUTS = 32               # default = 32 (VQGAN/DIFFUSION ONLY)
INPUT_IMAGE = ""        # path and filename of starting/input image, eg: samples/vectors/face_07.png
SKIP_STEPS = -1         # steps to skip when using init image (DIFFUSION ONLY)
LEARNING_RATE = 0.1     # default = 0.1 (VQGAN ONLY)
TRANSFORMER = ""        # needs to be a .yaml and .ckpt file in /checkpoints directory for whatever is specified here, default = vqgan_imagenet_f16_16384 (VQGAN ONLY)
CLIP_MODEL = ""         # default = ViT-B/32 (VQGAN ONLY)
OPTIMISER = ""          # default = Adam (VQGAN ONLY)
D_USE_VITB32 = "yes"    # load VitB32 CLIP model? (DIFFUSION ONLY)
D_USE_VITB16 = "yes"    # load VitB16 CLIP model? (DIFFUSION ONLY)
D_USE_VITL14 = "no"     # load VitL14 CLIP model? (DIFFUSION ONLY)
D_USE_RN101 = "no"      # load RN101 CLIP model? (DIFFUSION ONLY)
D_USE_RN50 = "yes"      # load RN50 CLIP model? (DIFFUSION ONLY)
D_USE_RN50x4 = "no"     # load RN50x4 CLIP model? (DIFFUSION ONLY)
D_USE_RN50x16 = "no"    # load RN50x16 CLIP model? (DIFFUSION ONLY)
D_USE_RN50x64 = "no"    # load RN50x64 CLIP model? (DIFFUSION ONLY)
STEPS = 50              # number of steps (STABLE DIFFUSION ONLY)
SCALE = 7.5             # guidance scale (STABLE DIFFUSION ONLY)
SAMPLES = 1             # number of samples to generate (STABLE DIFFUSION ONLY)
BATCH_SIZE = 1          # number of images to generate per sample (STABLE DIFFUSION ONLY)
STRENGTH = 0.75         # strength of starting image influence (STABLE DIFFUSION ONLY)
SD_LOW_MEMORY = "no"    # use the memory-optimized SD fork? (yes/no) (STABLE DIFFUSION ONLY)
SD_LOW_MEM_TURBO = "no" # faster at the cost of ~1GB VRAM (only when SD_LOW_MEMORY = "yes")
USE_UPSCALE = "no"      # upscale output images via ESRGAN/GFPGAN? (STABLE DIFFUSION ONLY)
UPSCALE_AMOUNT = 2.0    # amount to upscale, default is 2.0x (STABLE DIFFUSION ONLY)
UPSCALE_FACE_ENH = "no" # use GFPGAN, optimized for faces (STABLE DIFFUSION ONLY)
UPSCALE_KEEP_ORG = "no" # save the original non-upscaled image when using upscaling?
REPEAT = "no"           # when finished with all prompts, start back at beginning? (yes/no)

# Prevent threads from printing at same time.
print_lock = threading.Lock()

gpu_name = get_device_name()

# worker thread executes specified shell command
class Worker(threading.Thread):
    def __init__(self, command, do_upscale, upscl_amt, upscl_face, upscl_keep, callback=lambda: None):
        threading.Thread.__init__(self)
        self.command = command
        self.callback = callback
        self.use_upscale = do_upscale
        self.upscale_amount = upscl_amt
        self.upscale_face_enh = upscl_face
        self.upscale_keep_org = upscl_keep

    def run(self):
        # doing it this way in case the date has changed since the
        # work queue was created, vs having tons of files in a single dir
        self.command = self.command.replace("[[date]]", str(date.today()))
        sd = False

        # create output folder if it doesn't exist
        if " -o " in self.command:
            # this is vqgan/diffusion
            fullfilepath = self.command.split(" -o ",1)[1]
            filepath = fullfilepath.replace(fullfilepath[fullfilepath.rindex('/'):], "")
            Path(filepath).mkdir(parents=True, exist_ok=True)

            # check to see if output file already exists; find unique name if it does
            x = 1
            basefilepath = fullfilepath
            while exists(fullfilepath.replace('.png', '.jpg')):
                x += 1
                fullfilepath = basefilepath.replace(".png","") + '-' + str(x) + ".png"

            self.command = self.command.split(" -o ",1)[0] + " -o " + fullfilepath
        else:
            # this is stable diffusion
            sd = True
            # fullfilepath in the case of SD will simply be the output path since
            # SD doesn't support specifying input files
            fullfilepath = self.command.split(" --outdir ",1)[1]
            fullfilepath = fullfilepath.replace("../","")

        do_upscale = False
        face_enh = False
        upscale_keep_orig = False
        if self.use_upscale.lower() == "yes":
            do_upscale = True
            if self.upscale_face_enh.lower() == "yes":
                face_enh = True
            if self.upscale_keep_org.lower() == "yes":
                upscale_keep_orig = True

        with print_lock:
            print("Command: " + self.command)

        start_time = time.time()
        # invoke specified AI art process
        if not sd:
            subprocess.call(shlex.split(self.command))
        else:
            if sys.platform == "win32" or os.name == 'nt':
                subprocess.call(shlex.split(self.command), cwd=(cwd + '\stable-diffusion'))
            else:
                subprocess.call(shlex.split(self.command), cwd=(cwd + '/stable-diffusion'))

            gpu_id = '0'
            if '--device' in self.command:
                temp = self.command.split('--device', 1)[1]
                temp = temp.split(' --', 1)[0]
                gpu_id = temp.strip()

            samples_dir = fullfilepath + "/gpu_" + gpu_id

            if do_upscale:
                new_files = os.listdir(samples_dir)
                if len(new_files) > 0:
                    upscale(self.upscale_amount, samples_dir, gpu_id, face_enh)

                    # remove originals if upscaled version present
                    new_files = os.listdir(samples_dir)
                    for f in new_files:
                        if (".png" in f):
                            basef = f.replace(".png", "")
                            if basef[-2:] == "_u":
                                # this is an upscaled image, delete the original
                                # or save it in /original if desired
                                if exists(os.path.join(samples_dir, basef[:-2] + ".png")):
                                    if upscale_keep_orig:
                                        # move the original to /original
                                        orig_dir = os.path.join(fullfilepath + "original")
                                        Path(orig_dir).mkdir(parents=True, exist_ok=True)
                                        os.replace(os.path.join(samples_dir, basef[:-2] + ".png"), \
                                            os.path.join(orig_dir, basef[:-2] + ".png"))
                                    else:
                                        os.remove(os.path.join(samples_dir, basef[:-2] + ".png"))

            # find the new image(s) that SD created: re-name, process, and move them
            new_files = os.listdir(samples_dir)
            nf_count = 0
            exec_time = time.time() - start_time
            for f in new_files:
                if (".png" in f):
                    # todo: this is mostly a lazy copy from below and should be made into a function

                    # save just the essential prompt params to metadata
                    meta_prompt = self.command.split(" --prompt ",1)[1]
                    meta_prompt = meta_prompt.split(" --outdir ",1)[0]

                    if 'seed_' in f:
                        # grab seed from filename
                        actual_seed = f.replace('seed_', '')
                        actual_seed = actual_seed.split('_',1)[0]

                        # replace the seed in the command with the actual seed used
                        pleft = meta_prompt.split(" --seed ",1)[0]
                        pright = meta_prompt.split(" --seed ",1)[1].strip()
                        meta_prompt = pleft + " --seed " + actual_seed

                    upscale_text = ""
                    if do_upscale:
                        upscale_text = " (upscaled "
                        upscale_text += str(self.upscale_amount) + "x via "
                        if face_enh:
                            upscale_text += "GFPGAN)"
                        else:
                            upscale_text += "ESRGAN)"

                    pngImage = PngImageFile(samples_dir + '/' + f)
                    im = pngImage.convert('RGB')
                    exif = im.getexif()
                    exif[0x9286] = meta_prompt
                    exif[0x9c9c] = meta_prompt.encode('utf16')
                    exif[0x9c9d] = ('AI art (' + gpu_name + ')' + upscale_text).encode('utf16')
                    exif[0x0131] = "https://github.com/rbbrdckybk/ai-art-generator"
                    newfilename = dt.now().strftime('%Y%m-%d%H-%M%S-') + str(nf_count)
                    nf_count += 1
                    im.save(os.path.join(fullfilepath, newfilename + ".jpg"), exif=exif, quality=88)
                    if exists(os.path.join(samples_dir, f)):
                        os.remove(os.path.join(samples_dir, f))

            # remove the /samples dir if empty
            try:
                os.rmdir(samples_dir)
            except OSError as e:
                # nothing to do here, we only want to remove the dir
                # if it's completely empty
                pass

            fullfilepath = ""

        # save generation details as exif metadata for VQGAN and CLIP-guided diffusion outputs
        if exists(fullfilepath):
            exec_time = time.time() - start_time
            pngImage = PngImageFile(fullfilepath)

            # convert to jpg and remove the original png file
            im = pngImage.convert('RGB')
            exif = im.getexif()
            # usercomments
            exif[0x9286] = self.command
            # comments used by windows
            exif[0x9c9c] = self.command.encode('utf16')
            # author used by windows
            exif[0x9c9d] = gpu_name.encode('utf16')
            # software name used by windows
            exif[0x0131] = "AI Art (generated in " + str(datetime.timedelta(seconds=round(exec_time))) + ")"

            im.save(fullfilepath.replace('.png', '.jpg'), exif=exif, quality=88)
            if exists(fullfilepath.replace('.png', '.jpg')):
                os.remove(fullfilepath)

        with print_lock:
            print("Worker done.")
        self.callback()

# ESRGAN/GFPGAN upscaling:
# scale - upscale by this amount, default is 2.0x
# dir - upscale all images in this folder
# do_face_enhance - True/False use GFPGAN (for faces)
def upscale(scale, dir, gpu_id, do_face_enhance):
    command = "python inference_realesrgan.py -n RealESRGAN_x4plus --suffix u -s "

    # check that scale is a valid float, otherwise use default scale of 4
    try :
        float(scale)
        command += str(scale)
    except :
        command += "2"

    # specify gpu
    command += " -g " + gpu_id

    # append the input/output dir
    command += " -i ../" + dir + " -o ../" + dir

    # whether to use GFPGAN for faces
    if do_face_enhance:
        command += " --face_enhance"

    cwd = os.getcwd()
    print ("Invoking Real-ESRGAN: " + command)

    # invoke Real-ESRGAN
    if sys.platform == "win32" or os.name == 'nt':
        #subprocess.call(shlex.split(command), cwd=(cwd + '\Real-ESRGAN'), stderr=subprocess.DEVNULL)
        subprocess.call(shlex.split(command), cwd=(cwd + '\Real-ESRGAN'))
    else:
        subprocess.call(shlex.split(command), cwd=(cwd + '/Real-ESRGAN'), stderr=subprocess.DEVNULL)


# controller manages worker thread(s) and user input
# TODO change worker_idle to array of bools to manage multiple threads/gpus
class Controller:
    def __init__(self, prompt_file):

        self.process = PROCESS
        self.width = WIDTH
        self.height = HEIGHT
        self.iterations = ITERATIONS
        self.seed = SEED
        self.cuda_device = CUDA_DEVICE
        self.learning_rate = LEARNING_RATE
        self.cuts = CUTS
        self.input_image = INPUT_IMAGE
        self.skip_steps = SKIP_STEPS
        self.transformer = TRANSFORMER
        self.clip_model = CLIP_MODEL
        self.optimiser = OPTIMISER
        self.d_use_vitb32 = D_USE_VITB32
        self.d_use_vitb16 = D_USE_VITB16
        self.d_use_vitl14 = D_USE_VITL14
        self.d_use_rn101 = D_USE_RN101
        self.d_use_rn50 = D_USE_RN50
        self.d_use_rn50x4 = D_USE_RN50x4
        self.d_use_rn50x16 = D_USE_RN50x16
        self.d_use_rn50x64 = D_USE_RN50x64
        self.steps = STEPS
        self.scale = SCALE
        self.samples = SAMPLES
        self.batch_size = BATCH_SIZE
        self.strength = STRENGTH
        self.sd_low_memory = SD_LOW_MEMORY
        self.sd_low_mem_turbo = SD_LOW_MEM_TURBO
        self.use_upscale = USE_UPSCALE
        self.upscale_amount = UPSCALE_AMOUNT
        self.upscale_face_enh = UPSCALE_FACE_ENH
        self.upscale_keep_org = UPSCALE_KEEP_ORG
        self.repeat = REPEAT
        self.work_queue = deque()
        self.work_done = False
        self.worker_idle = True
        self.is_paused = False
        self.jobs_done = 0

        # text file containing all of the prompt/style/etc info
        self.prompt_file_name = prompt_file

        # lists for prompts/styles
        self.subjects = list()
        self.styles = list()
        self.prefixes = list()
        self.suffixes = list()

        self.__init_lists(self.subjects, "subjects")
        self.__init_lists(self.styles, "styles")
        self.__init_lists(self.prefixes, "prefixes")
        self.__init_lists(self.suffixes, "suffixes")

        if sys.platform == "win32" or os.name == 'nt':
            #keyboard.on_press_key("f10", lambda _:self.pause_callback())
            #keyboard.on_press_key("f9", lambda _:self.exit_callback())
            keyboard.add_hotkey("ctrl+shift+p", lambda: self.pause_callback())
            keyboard.add_hotkey("ctrl+shift+q", lambda: self.exit_callback())
            keyboard.add_hotkey("ctrl+shift+r", lambda: self.reload_callback())

        self.init_work_queue()
        with print_lock:
            print("Queued " + str(len(self.work_queue)) + " work items from " + self.prompt_file_name + ".")

    # init the lists
    def __init_lists(self, which_list, search_text):
        with open(self.prompt_file_name) as f:
            lines = f.readlines()

            search_header = '[' + search_text + ']'
            found_header = False

            # find the search text and read until the next search header
            for line in lines:
                # ignore comments and strip whitespace
                line = line.strip().split('#', 1)
                line = line[0].strip()

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

    # returns a random prefix from the prompt file
    def prefix(self):
        prefix = ''
        if len(self.prefixes) > 0:
            x = random.randint(0, len(self.prefixes)-1)
            prefix = self.prefixes[x]
        return prefix

    # returns a random suffix from the prompt file
    def suffix(self):
        suffix = ''
        if len(self.suffixes) > 0:
            x = random.randint(0, len(self.suffixes)-1)
            suffix = self.suffixes[x]
        return suffix

    # build a work queue with the specified prompt and style files
    def init_work_queue(self):

        # construct work queue consisting of all prompt+style combos
        for subject in self.subjects:

            # if this is a setting directive, handle it
            if subject[0] == '!':
                self.change_setting(subject)

            # otherwise build the command
            else:
                base = ""

                if self.process == "stablediff":

                    if self.input_image != "":
                        base = "python scripts_mod/img2img.py"
                        if self.sd_low_memory.lower() == "yes":
                            base = "python scripts_mod/optimized_img2img.py"

                    else:
                        base = "python scripts_mod/txt2img.py"
                        if self.sd_low_memory.lower() == "yes":
                            base = "python scripts_mod/optimized_txt2img.py"

                        base += " --W " + str(self.width) \
                            + " --H " + str(self.height)

                    # additional common params
                    if int(self.cuda_device) > 0:
                        base += " --device \"cuda:" + str(self.cuda_device) + "\""
                    if self.sd_low_memory.lower() == "yes" and self.sd_low_mem_turbo.lower() == "yes":
                        base += " --turbo"
                    base += " --skip_grid" \
                        + " --n_iter " + str(self.samples) \
                        + " --n_samples " + str(self.batch_size) \
                        + " --prompt \""

                else:
                    # vqgan & diffusion shared initial setup
                    base = "python " + self.process + ".py" \
                        + " -s " + str(self.width) + " " + str(self.height) \
                        + " -i " + str(self.iterations) \
                        + " -cuts " + str(self.cuts) \
                        + " -p \""

                input_name = self.prompt_file_name.split('/')
                input_name = input_name[len(input_name)-1]
                input_name = input_name.split('\\')
                input_name = input_name[len(input_name)-1]
                outdir="output/[[date]]" + '-' + slugify(input_name.split('.', 1)[0])

                # quick fix for empty/non-existant styles
                if len(self.styles) == 0:
                    self.styles.append("")

                # queue a work item for each style/artist
                for style in self.styles:
                    if self.process == "stablediff":
                        # order matters more in stable diffusion, get the style in front of suffix
                        work = base + (self.prefix() + " " + subject + ", " + style.strip() + ", " + self.suffix()).strip() + "\""
                    else:
                        base += (self.prefix() + ' ' + subject + ' ' + self.suffix()).strip()
                        work = base + " | " + style.strip() + "\""

                    # VQGAN+CLIP -specific params
                    if self.process == "vqgan":
                        work += " -lr " + str(self.learning_rate)

                        if self.transformer != "":
                            work += " -conf checkpoints/" + self.transformer + ".yaml -ckpt checkpoints/" + self.transformer + ".ckpt"
                        if self.clip_model != "":
                            work += " -m " + self.clip_model
                        if self.optimiser != "":
                            work += " -opt " + self.optimiser
                        if self.cuda_device != "":
                            work += " -cd \"cuda:" + str(self.cuda_device) + "\""

                    # CLIP-guided diffusion -specific params:
                    if self.process == "diffusion":
                        work += " -cd " + str(self.cuda_device)
                        work += " -dvitb32 " + self.d_use_vitb32
                        work += " -dvitb16 " + self.d_use_vitb16
                        work += " -dvitl14 " + self.d_use_vitl14
                        work += " -drn101 " + self.d_use_rn101
                        work += " -drn50 " + self.d_use_rn50
                        work += " -drn50x4 " + self.d_use_rn50x4
                        work += " -drn50x16 " + self.d_use_rn50x16
                        work += " -drn50x64 " + self.d_use_rn50x64

                    seed = random.randint(1, 2**32) - 1000
                    if int(self.seed) > -1:
                        seed = int(self.seed)

                    # Stable Diffusion -specific params:
                    if self.process == "stablediff":
                        # Stable Diffusion -specific closing args:
                        if self.input_image != "":
                            work += " --init-img \"../" + self.input_image + "\"" + " --strength " + str(self.strength)
                        work += " --ddim_steps " + str(self.steps) \
                            + " --scale " + str(self.scale) \
                            + " --seed " + str(seed) \
                            + " --outdir ../" + outdir

                    else:
                        # vqgan and diffusion -shared closing args:
                        if self.input_image != "":
                            work += " -ii \"" + self.input_image + "\""
                            if self.process == "diffusion" and int(self.skip_steps) > -1:
                                work += " -ss " + self.skip_steps

                        name_subj = slugify(subject)
                        name_subj = re.sub(":[-+]?\d*\.?\d+|[-+]?\d+", "", name_subj)
                        name_style = slugify(style)
                        name_style = re.sub(":[-+]?\d*\.?\d+|[-+]?\d+", "", name_style)
                        if len(name_subj) > (180 - len(name_style)):
                            x = 180 - len(name_style)
                            name_subj = name_subj[0:x]

                        work += " -sd " + str(seed) + " -o " + outdir + "/" + name_subj + '-' + name_style + ".png"

                    # work args built, add to queue
                    self.work_queue.append(work)

    # handle whatever settings directives that are allowed in the prompt file here
    def change_setting(self, setting_string):
        ss = re.search('!(.+?)=', setting_string)
        if ss:
            command = ss.group(1).lower().strip()
            value = setting_string.split("=",1)[1].strip()

            # python switch
            if command == 'process':
                if value == '':
                    value = PROCESS
                self.process = value

            elif command == 'cuda_device':
                if value == '':
                    value = CUDA_DEVICE
                self.cuda_device = value

            elif command == 'width':
                if value == '':
                    value = WIDTH
                self.width = value

            elif command == 'height':
                if value == '':
                    value = HEIGHT
                self.height = value

            elif command == 'iterations':
                if value == '':
                    value = ITERATIONS
                self.iterations = value

            elif command == 'seed':
                if value == '':
                    value = SEED
                self.seed = value

            elif command == 'learning_rate':
                if value == '':
                    value = LEARNING_RATE
                self.learning_rate = value

            elif command == 'cuts':
                if value == '':
                    value = CUTS
                self.cuts = value

            elif command == 'input_image':
                self.input_image = value

            elif command == 'skip_steps':
                if value == '':
                    value = SKIP_STEPS
                self.skip_steps = value

            elif command == 'transformer':
                if value == 'vqgan_imagenet_f16_16384':
                    value = ''
                self.transformer = value

            elif command == 'clip_model':
                self.clip_model = value

            elif command == 'optimiser':
                self.optimiser = value

            elif command == 'd_vitb32':
                self.d_use_vitb32 = value

            elif command == 'd_vitb16':
                self.d_use_vitb16 = value

            elif command == 'd_vitl14':
                self.d_use_vitl14 = value

            elif command == 'd_rn101':
                self.d_use_rn101 = value

            elif command == 'd_rn50':
                self.d_use_rn50 = value

            elif command == 'd_rn50x4':
                self.d_use_rn50x4 = value

            elif command == 'd_rn50x16':
                self.d_use_rn50x16 = value

            elif command == 'd_rn50x64':
                self.d_use_rn50x64 = value

            elif command == 'steps':
                if value == '':
                    value = STEPS
                self.steps = value

            elif command == 'scale':
                if value == '':
                    value = SCALE
                self.scale = value

            elif command == 'samples':
                if value == '':
                    value = SAMPLES
                self.samples = value

            elif command == 'batch_size':
                if value == '':
                    value = BATCH_SIZE
                self.batch_size = value

            elif command == 'strength':
                if value == '':
                    value = STRENGTH
                self.strength = value

            elif command == 'sd_low_memory':
                if value == '':
                    value = SD_LOW_MEMORY
                self.sd_low_memory = value

            elif command == 'sd_low_mem_turbo':
                if value == '':
                    value = SD_LOW_MEM_TURBO
                self.sd_low_mem_turbo = value

            elif command == 'use_upscale':
                if value == '':
                    value = USE_UPSCALE
                self.use_upscale = value

            elif command == 'upscale_amount':
                if value == '':
                    value = UPSCALE_AMOUNT
                self.upscale_amount = value

            elif command == 'upscale_face_enh':
                if value == '':
                    value = UPSCALE_FACE_ENH
                self.upscale_face_enh = value

            elif command == 'upscale_keep_org':
                if value == '':
                    value = UPSCALE_KEEP_ORG
                self.upscale_keep_org = value

            elif command == 'repeat':
                if value == '':
                    value = REPEAT
                self.repeat = value

            else:
                print("\n*** WARNING: prompt file command not recognized: " + command.upper() + " (it will be ignored!) ***\n")
                time.sleep(1.5)

    # start a new worker thread
    def do_work(self, command):
        self.worker_idle = False
        with print_lock:
            print("\n\nWorker starting job #" + str(self.jobs_done+1) + ":")
        thread = Worker(command, \
            self.use_upscale, \
            self.upscale_amount, \
            self.upscale_face_enh, \
            self.upscale_keep_org, \
            self.on_work_done)
        thread.start()

    # callback for worker threads when finished
    def on_work_done(self):
        self.worker_idle = True
        self.jobs_done += 1

    # pause execution at user request
    def pause_callback(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            with print_lock:
                print("\n\n*** Work will be paused when current operation finishes! ***")
                print("*** (press 'CTRL+SHIFT+P' again to unpause, or 'CTRL+SHIFT+Q' to quit) ***\n")
        else:
            with print_lock:
                print("\n*** Work resuming! ***\n")

    # allow exit at user request if currently paused
    def exit_callback(self):
        if self.is_paused:
            print("Exiting...")
            self.work_done = True

    # discards the current work queue and re-builds it from the prompt file
    # useful if the file has changed and the user wants to reload it
    def reload_callback(self):
        with print_lock:
            print("\n\n*** Discarding current work queue and re-building! ***")

        self.work_queue = deque()
        self.subjects = list()
        self.styles = list()
        self.prefixes = list()
        self.suffixes = list()
        self.__init_lists(self.subjects, "subjects")
        self.__init_lists(self.styles, "styles")
        self.__init_lists(self.prefixes, "prefixes")
        self.__init_lists(self.suffixes, "suffixes")
        self.init_work_queue()

        with print_lock:
            print("*** Queued " + str(len(self.work_queue)) + " work items from " + self.prompt_file_name + "! ***")


# for easy reading of prompt/style files
class TextFile():
    def __init__(self, filename):
        self.lines = deque()

        with open(filename) as f:
            l = f.readlines()

        for x in l:
            x = x.strip();
            if x != "" and x[0] != '#':
                # these lines are actual prompts
                x = x.strip('\n')
                self.lines.append(x)

    def next_line(self):
        return self.lines.popleft()

    def lines_remaining(self):
        return len(self.lines)

# Taken from https://github.com/django/django/blob/master/django/utils/text.py
# Using here to make filesystem-safe directory names
def slugify(value, allow_unicode=False):
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    value = re.sub(r'[-\s]+', '-', value).strip('-_')
    # added in case of very long filenames due to multiple prompts
    return value[0:180]

# entry point
if __name__ == '__main__':

    if len(sys.argv) > 1:
        prompt_filename = sys.argv[1]
        if not exists(prompt_filename):
            print("\nThe specified prompt file '" + prompt_filename + "' doesn't exist!")
            print("Please specify a valid text file containing your prompt information.")
            exit()

        control = Controller(prompt_filename)
        passes = 0

        # main work loop
        while not control.work_done:
            # worker is idle, start some work
            if (control.worker_idle and not control.is_paused):
                if len(control.work_queue) > 0:
                    # get a new prompt or setting directive from the queue
                    new_work = control.work_queue.popleft()
                    control.do_work(new_work)
                else:
                    # no more prompts to work on
                    passes += 1
                    if (passes > 1):
                        print('\nAll work done (completed ' + str(passes) + ' runs through prompt file)!')
                    else:
                        print('\nAll work done!')

                    if control.repeat.lower() == 'yes':
                        print('Restarting from top of prompt file!')
                        control = Controller(prompt_filename)
                    else:
                        control.work_done = True
            else:
                time.sleep(.01)

    else:
        print("\nUsage: python make_art.py [prompt file]")
        print("Example: python make_art.py prompts.txt")
        exit()

    if control and control.jobs_done > 0:
        print("Total jobs done: " + str(control.jobs_done))
    exit()
