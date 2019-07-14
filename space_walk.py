# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Script for generating a smooth movie following a closed curve in latent space"""

import os
import sys
import pickle
import numpy as np
import PIL.Image
import dnnlib.tflib as tflib
import shutil
from argparse import ArgumentParser
import tqdm


def build_parser():
  parser = ArgumentParser()
  parser.add_argument('--seed',
                      help='starting seed for the random number generator',
                      default=None,
                      dest='seed',
                      type=int,
                      required=False)
  parser.add_argument('--output-dir',
                      dest='output_dir',
                      help='directory of output images and movies',
                      default='results',
                      required=False)
  parser.add_argument('--num_steps',
                      dest='steps',
                      default=2000,
                      type=int,
                      help='number of steps or frames',
                      required=False)
  parser.add_argument('--radius',
                      dest='radius',
                      help='radius of the sphere the point traces in latent space',
                      type=float,
                      default=1.0,
                      required=False)
  return parser

def get_random(init_seed):
  r = np.random.RandomState()
  if init_seed is None:
    actual_seed = r.randint(2**32 - 1)
    r.seed(actual_seed)
  else:
    actual_seed = init_seed

  return actual_seed, r

def make_movies():
  opts = build_parser().parse_args(args=sys.argv[1:])

  # Initialize TensorFlow.
  tflib.init_tf()
  url = 'results/00001-sgan-evan-1024-2gpu/network-snapshot-025000.pkl'
  with open(url, "rb") as f:
    _G, _D, Gs = pickle.load(f)
    # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
    # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
    # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

  while True:
    make_movie(Gs, opts)

def make_movie(Gs, options, seed=None):
  output_dir = options.output_dir
  os.makedirs(output_dir, exist_ok=True)
  steps = options.steps
  radius = options.radius

  N = Gs.input_shape[1]

  seed, rnd = get_random(seed)

  period_array = rnd.randint(low=1, high=10, size=int(N / 2))
  centre = rnd.randn(1, N)
  theta_delta = 6.2831853072 / steps * 2

  mv_cmd = 'ffmpeg -framerate 25 -i {image_dir}/example%04d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {output_dir}/{seed}.mp4'

  image_dir = os.path.join(output_dir, str(seed))
  os.makedirs(image_dir)

  pbar = tqdm.tqdm(range(steps))
  for i in pbar:
    point = centre.copy()

    # Guaranteed periodic!
    for a in range(1, N, 2):
      p = period_array[int((a - 1) / 2)]
      point[0, a - 1] += radius * np.sin(p * i * theta_delta / 2) + 2 * radius * np.cos(p * i * theta_delta)
      point[0, a] += radius * np.cos(p * i * theta_delta / 2) + 2 * radius * np.sin(p * i * theta_delta)

    # Generate image.
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    images = Gs.run(point, None, truncation_psi=0.7, randomize_noise=False, output_transform=fmt)

    # Save image.
    png_filename = os.path.join(image_dir, 'example%04d.png' % i)
    PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

  rc = os.system(mv_cmd.format(seed=seed, output_dir=output_dir, image_dir=image_dir))

  if rc != 0:
    print('cmd failed rc={rc}'.format(rc=rc))
  else:
    shutil.rmtree(image_dir)

make_movies()