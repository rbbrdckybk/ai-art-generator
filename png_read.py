# simple utility for reading metadata back from created PNG files
# use this on your images to see exactly what VQGAN+CLIP settings were used to create it

import sys
from os.path import exists
from PIL import Image

if len(sys.argv) > 1:
    if exists(sys.argv[1]):
        filename = sys.argv[1]
        pngImage = Image.open(filename)
        pngImage.load()
        print(pngImage.info['VQGAN+CLIP'])
        pngImage.close()
    else:
        print("Error, specified file doesn't exist: " + sys.argv[1])
else:
    print("Usage: python png_read.py [filename]")
    print("Example: python png_read.py output/a-clown/pencil-sketch.png")
