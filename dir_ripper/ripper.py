from multiprocessing.pool import Pool
from multiprocessing import Manager
import os
from queue import Empty
from functools import reduce


def walk(path, processor, args = (), n_threads = 5, filter = lambda f: True):
  q = Manager().Queue()

  for fe in os.scandir(path):
    if fe.is_file() and filter(fe):
      q.put(fe.path)

  p = Pool(processes=n_threads)
  res = list(map(lambda _: p.apply_async(worker, (q, processor, args)), range(n_threads)))
  cnt = reduce(lambda r1, r2: r1+r2, map(lambda r: r.get(), res))

  return cnt


def worker(queue, processor, args):
  cnt = 0
  while True:
    try:
      fe = queue.get_nowait()
      processor(fe, *args)
      queue.task_done()
      cnt = cnt + 1
    except Empty:
      return cnt


def test_processor(p):
  pass


if __name__ == '__main__':
  path = '/home/evan/neurosim/images/inputs'
  cnt = walk(path, test_processor)
  print(cnt)
