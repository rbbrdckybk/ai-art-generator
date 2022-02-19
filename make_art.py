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
from datetime import date
from pathlib import Path
from collections import deque
from PIL.PngImagePlugin import PngImageFile, PngInfo
from torch.cuda import get_device_name

if sys.platform == "win32" or os.name == 'nt':
    import keyboard

# these can be overriden with prompt file directives, no need to change them here
PROCESS = "vqgan"       # which AI process to use, default is vqgan
WIDTH = 512             # output image width, default is 512
HEIGHT = 512            # output image height, default is 512
ITERATIONS = 500        # number of times to run, default is 500
CUTS = 32               # default = 32
INPUT_IMAGE = ""        # path and filename of starting image, eg: samples/vectors/face_07.png
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

# Prevent threads from printing at same time.
print_lock = threading.Lock()

gpu_name = get_device_name()

# worker thread executes specified shell command
class Worker(threading.Thread):
    def __init__(self, command, callback=lambda: None):
        threading.Thread.__init__(self)
        self.command = command
        self.callback = callback

    def run(self):
        # create output folder if it doesn't exist
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

        with print_lock:
            print("Command: " + self.command)

        start_time = time.time()
        # invoke specified AI art process
        subprocess.call(shlex.split(self.command))
        exec_time = time.time() - start_time

        # save generation details as exif metadata
        if exists(fullfilepath):
            pngImage = PngImageFile(fullfilepath)
            #metadata = PngInfo()
            #metadata.add_text("VQGAN+CLIP", self.command)
            #pngImage.save(fullfilepath, pnginfo=metadata)
            #pngImage = PngImageFile(fullfilepath)

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

# controller manages worker thread(s) and user input
# TODO change worker_idle to array of bools to manage multiple threads/gpus
class Controller:
    def __init__(self, prompt_file):

        self.process = PROCESS
        self.width = WIDTH
        self.height = HEIGHT
        self.iterations = ITERATIONS
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
                base = "python " + self.process + ".py" \
                    + " -s " + str(self.width) + " " + str(self.height) \
                    + " -i " + str(self.iterations) \
                    + " -cuts " + str(self.cuts) \
                    + " -p \""
                base += (self.prefix() + ' ' + subject + ' ' + self.suffix()).strip()

                input_name = self.prompt_file_name.split('/')
                input_name = input_name[len(input_name)-1]
                input_name = input_name.split('\\')
                input_name = input_name[len(input_name)-1]
                outdir="output/" + str(date.today()) + '-' + slugify(input_name.split('.', 1)[0])

                for style in self.styles:
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

                    # CLIP-guided diffusion -specific params:
                    if self.process == "diffusion":
                        work += " -dvitb32 " + self.d_use_vitb32
                        work += " -dvitb16 " + self.d_use_vitb16
                        work += " -dvitl14 " + self.d_use_vitl14
                        work += " -drn101 " + self.d_use_rn101
                        work += " -drn50 " + self.d_use_rn50
                        work += " -drn50x4 " + self.d_use_rn50x4
                        work += " -drn50x16 " + self.d_use_rn50x16
                        work += " -drn50x64 " + self.d_use_rn50x64

                    # common params
                    if self.input_image != "":
                        work += " -ii \"" + self.input_image + "\""
                        if self.process == "diffusion" and int(self.skip_steps) > -1:
                            work += " -ss " + self.skip_steps

                    #seed = random.randint(100000000000000,999999999999999)
                    seed = random.randint(1, 2**32) - 1
                    name_subj = slugify(subject)
                    name_subj = re.sub(":[-+]?\d*\.?\d+|[-+]?\d+", "", name_subj)
                    name_style = slugify(style)
                    name_style = re.sub(":[-+]?\d*\.?\d+|[-+]?\d+", "", name_style)
                    if len(name_subj) > (180 - len(name_style)):
                        x = 180 - len(name_style)
                        name_subj = name_subj[0:x]

                    work += " -sd " + str(seed) + " -o " + outdir + "/" + name_subj + '-' + name_style + ".png"

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

            else:
                print("\n*** WARNING: prompt file command not recognized: " + command.upper() + " (it will be ignored!) ***\n")
                time.sleep(1.5)

    # start a new worker thread
    def do_work(self, command):
        self.worker_idle = False
        with print_lock:
            print("\n\nWorker starting job #" + str(self.jobs_done+1) + ":")
        thread = Worker(command, self.on_work_done)
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
                    print('\nAll work done!')
                    control.work_done = True
            else:
                time.sleep(.01)

    else:
        print("\nUsage: python make_art.py [prompt file]")
        print("Example: python make_art.py prompts.txt")

    if control and control.jobs_done > 0:
        print("Total jobs done: " + str(control.jobs_done))
    exit()
