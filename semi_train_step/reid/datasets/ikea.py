from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np
import os
import errno

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import read_json
from ..utils.serialization import write_json





class Ikea(Dataset):
  def __init__(self, root, split_id=0, num_val=100, download=True):
    super(Ikea, self).__init__(root, split_id=split_id)
    self.root = root

    self.load()
  


  ########################  
  # Added
  def load(self, verbose=True):
    import re
    from glob import glob
    import shutil
    #train set
    image_dir = osp.join(self.root, 'images')
    image_list = os.listdir(image_dir)
    ret = []
    for image in image_list:
      #name = image.split('.')[0]
      ret.append((image, 0, 0))
    self.trainval = ret

    #test set
    test_dir = osp.join(self.root, 'test')
    try:
      os.makedirs(test_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  

    identities = []
    all_pids = {}
    exdir = '/home/famu/jh/SSG_ikea/dataset/ikea'
    def register(subdir, pattern=re.compile(r'([-\d]+)_c(\d)')):
        fnames = [] ###### New Add. Names of images in new dir 
        fpaths = sorted(glob(osp.join(exdir, subdir, '*.jpg')))
        pids = set()
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            assert 1 <= cam <= 8
            cam -= 1
            if pid not in all_pids:
                all_pids[pid] = len(all_pids)
            pid = all_pids[pid]
            pids.add(pid)
            if pid >= len(identities):
                assert pid == len(identities)
                identities.append([[] for _ in range(8)])  # 8 camera views
            fname = ('{:08d}_{:02d}_{:04d}.jpg'
                      .format(pid, cam, len(identities[pid][cam])))
            identities[pid][cam].append(fname)
            shutil.copy(fpath, osp.join(test_dir, fname))
            fnames.append(fname)######## added
        return pids, fnames

    gallery_pids, gallery_fnames = register('gallery')
    query_pids, query_fnames = register('query')
    assert query_pids <= gallery_pids

    # Save meta information into a json file
    meta = {'name': 'DukeMTMC', 'shot': 'multiple', 'num_cameras': 8,
            'identities': identities,
            'query_fnames': query_fnames,########## Added
            'gallery_fnames': gallery_fnames} ######### Added
    write_json(meta, osp.join(self.root, 'meta.json'))

    splits = [{
            'query': sorted(list(query_pids)),
            'gallery': sorted(list(gallery_pids))}]
    write_json(splits, osp.join(self.root, 'splits.json'))

    splits = read_json(osp.join(self.root, 'splits.json'))
    if self.split_id >= len(splits):
        raise ValueError("split_id exceeds total splits {}"
                                          .format(len(splits)))
    self.split = splits[self.split_id]
    
    #gallery set
    self.meta = read_json(osp.join(self.root, 'meta.json'))
    query_fnames = self.meta['query_fnames']
    gallery_fnames = self.meta['gallery_fnames']
    self.query = []
    for fname in query_fnames:
        name = osp.splitext(fname)[0]
        pid, cam, _ = map(int, name.split('_'))
        self.query.append((fname, pid, cam))
    self.gallery = []
    for fname in gallery_fnames:
        name = osp.splitext(fname)[0]
        pid, cam, _ = map(int, name.split('_'))
        self.gallery.append((fname, pid, cam))





    
    ##########

    if verbose:
      print(self.__class__.__name__, "dataset loaded")
      print("  subset   | # ids | # images")
      print("  ---------------------------")
      print("  trainval | {:5d} | {:8d}"
            .format(self.num_trainval_ids, len(self.trainval)))
      print("  query    | {:5d} | {:8d}"
            .format(len(self.split['query']), len(self.query)))
      print("  gallery  | {:5d} | {:8d}"
            .format(len(self.split['gallery']), len(self.gallery)))
      
  ########################