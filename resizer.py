import os
import PIL
from PIL import Image
from argparse import ArgumentParser
from dir_ripper import ripper

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


def resize_crop(img, resize_to):

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


def check_resize(img_nm, resize_to, out_dir):
  out_img_nm = os.path.join(out_dir, os.path.basename(img_nm))
  if os.path.exists(out_img_nm):
    return

  img = Image.open(img_nm)
  width, height = img.size
  if width<resize_to or height<resize_to:
    print('Input ' + img_nm + ' is too small. ' + str(width) + 'x' + str(height))
    return

  out_img = resize_crop(img, resize_to)
  out_img.save(out_img_nm)


def create_if_needed(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def check_size(fn, output_dir):
  try:
    img = Image.open(fn)
  except OSError as ex:
    print(ex)
    return

  width, height = img.size
  if width <= 1024 and height <= 1024:
    new_name = os.path.join(output_dir, os.path.basename(fn))
    print(new_name)
    # os.rename(fn, new_name)


def main(argv):
    parser = build_parser()
    options = parser.parse_args(args=argv)

    input_dir = options.input_dir
    output_dir = options.output_dir
    create_if_needed(output_dir)
    size = int(options.size)

    ripper.walk(input_dir, check_resize, args = (size, output_dir))
    # ripper.walk(input_dir, check_size, args = (output_dir,))


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])


