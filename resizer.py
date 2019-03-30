import os
import PIL
from PIL import Image
from argparse import ArgumentParser


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--input-dir',
                        dest='input_dir',
                        help='directory of input images',
                        required=True)
    parser.add_argument('--output-dir',
                        dest='output_dir',
                        help='directory of output images',
                        required=True)
    parser.add_argument('--size',
                        dest='size',
                        help='size of the longest edge',
                        required=True)
    return parser


def resize(img, resize_to):
    img = Image.open(img)

    width, height = img.size

    # square it
    if width > height:
        delta = width - height
        left = int(delta / 2)
        upper = 0
        right = height + left
        lower = height
    else:
        delta = height - width
        left = 0
        upper = int(delta / 2)
        right = width
        lower = width + upper

    img = img.crop((left, upper, right, lower))

    return img.resize((resize_to, resize_to), PIL.Image.ANTIALIAS)


def list_files(input_dir):
    return [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith('.jpg')]


def create_if_needed(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def main(argv):
    parser = build_parser()
    options = parser.parse_args(args=argv)

    input_dir = options.input_dir
    input_files = list_files(input_dir)
    output_dir = options.output_dir
    create_if_needed(output_dir)
    size = int(options.size)

    for in_img in input_files:
        out_img = resize(os.path.join(input_dir, in_img), size)
        out_img.save(os.path.join(output_dir, in_img))


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])


