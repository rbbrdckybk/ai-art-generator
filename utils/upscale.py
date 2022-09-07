# Copyright 2021 - 2022, Bill Kennedy (https://github.com/rbbrdckybk/ai-art-generator)
# SPDX-License-Identifier: MIT

# upscaling tool
# Point this at a directory full of images created via AI Art Generator to generate
# upscaled (via ESRGAN/GFPGAN) versions. All of the embedded metadata from the
# original images will be preserved in the upscaled version.

import argparse
import shlex
import subprocess
import sys
import unicodedata
import re
import random
import os
from os.path import exists
from pathlib import Path
from PIL import Image

#import warnings
#warnings.filterwarnings("ignore")

def upscale(in_dir, scale, do_face_enhance, do_single):
    print("\nStarting...")
    command = "python inference_realesrgan.py -n RealESRGAN_x4plus --suffix u -s "

    # check that scale is a valid float, otherwise use default scale of 2
    try :
        float(scale)
        command += str(scale)
    except :
        command += "2"

    # append the input/output dir
    out_dir = in_dir + "\\upscaled"
    command += " -i \"" + in_dir + "\" -o \"" + out_dir + "\""

    # whether to use GFPGAN for faces
    if do_face_enhance:
        command += " --face_enhance"

    # make output dir if it doesn't exist
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd().replace('utils', '')
    esrgan_dir = cwd + 'Real-ESRGAN'

    # invoke Real-ESRGAN
    if not do_single:
        print ("Invoking Real-ESRGAN: " + command)
        subprocess.call(command, cwd=(esrgan_dir), stderr=subprocess.DEVNULL)
    else:
        # we're going to call ESRGAN once per image
        orig_command = command
        new_files = os.listdir(in_dir)
        for f in new_files:
            file_ext = f[-4:]
            if (file_ext == ".jpg") or (file_ext == ".png"):
                # this is an image, let's try to upscale it
                command = orig_command.replace(" -i \"" + in_dir, " -i \"" + in_dir + "\\" + f)
                print ("Invoking Real-ESRGAN (per image): " + command)
                subprocess.call(command, cwd=(esrgan_dir), stderr=subprocess.DEVNULL)


    # create text string to append to metadata
    upscale_text = " (upscaled "
    upscale_text += str(scale) + "x via "
    if do_face_enhance:
        upscale_text += "GFPGAN)"
    else:
        upscale_text += "ESRGAN)"

    # copy metadata from original files to upscaled files
    print('\nCopying metadata from original files to upscaled files...')
    new_files = os.listdir(out_dir)
    for f in new_files:
        f = f.lower()
        file_ext = f[-4:]
        if (file_ext == ".jpg") or (file_ext == ".png"):
            basef = f.replace(file_ext, "")
            if basef[-2:] == "_u":
                # this is an upscaled image,
                # check for an original w/ metadata
                original_file = in_dir + "/" + basef[:-2] + file_ext
                if exists(original_file):
                    # found the corresponding original,
                    # copy the metadata
                    orig_Image = Image.open(original_file)
                    im = orig_Image.convert('RGB')
                    exif = im.getexif()

                    # only modify the upscale exif if upscale factor > 1.0
                    if scale > 1:
                        try:
                            exif[0x9c9d] = exif[0x9c9d] + upscale_text.encode('utf16')
                        except KeyError as e:
                            exif[0x9c9d] = upscale_text.encode('utf16')

                    # save to the new image
                    new_Image = Image.open(out_dir + "/" + f)
                    im = new_Image.convert('RGB')
                    im.save(out_dir + "/" + basef[:-2] + file_ext, exif=exif, quality=88)

                    # remove the upscaled file with the '_u' extension
                    if exists(out_dir + "/" + basef + file_ext):
                        os.remove(out_dir + "/" + basef + file_ext)

    print('All done!\n')

def main():
    print('\nUpscaling (' + str(UPSCALE_AMOUNT) + 'x) all .jpg images in: ' + os.getcwd() + '\\' + UPSCALE_IN_DIR)
    print('Upscaled images will be written to: ' + os.getcwd() + '\\' + UPSCALE_OUT_DIR + '\n')
    print
    upscale(UPSCALE_AMOUNT, UPSCALE_IN_DIR, UPSCALE_OUT_DIR, UPSCALE_FACE_ENH)


# entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--imgdir",
        type=str,
        default="",
        help="the directory containing images"
    )

    parser.add_argument(
        "--amount",
        type=float,
        nargs="?",
        default="2.0",
        help="amount to upscale (default = 2.0)"
    )

    parser.add_argument(
        "--faces",
        action='store_true',
        help="use face enhancement (GFPGAN)"
    )

    parser.add_argument(
        "--single",
        action='store_true',
        help="invoke ESRGAN once per image? (slower, try if you get out of memory errors)"
    )

    opt = parser.parse_args()

    if opt.imgdir != "":
        if not os.path.exists(opt.imgdir):
            print("\nThe specified path '" + opt.imgdir + "' doesn't exist!")
            print("Please specify a valid directory containing images.")
            exit()
        else:
            upscale(opt.imgdir, opt.amount, opt.faces, opt.single)

    else:
        print("\nUsage: python upscale.py --imgdir [directory containing images]")
        print("Example: python upscale.py --imgdir \"c:\images\"")
        print("Options/help: python upscale.py --help")
        exit()
