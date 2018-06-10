from __future__ import print_function

import os
import numpy as np
import sys
import progressbar
import random

class fdata(object):

    def __init__(self):
        self.fmaps_list = []
        self.fmaps_attr_list = []

    def make_fmaps(self,filterbanks):
        # Init Indexes
        map = [n for n in range(0,len(filterbanks))]
        map_h = [0 for n in range(0,8)]
        map_t = [len(filterbanks)-1 for n in range(0,8)]
        map = np.append(map_h,map)
        map = np.append(map,map_t)

        # Create Fmaps
        fmaps = []

        for i in range(0,len(filterbanks)):
            fmap = np.stack([filterbanks[n] for n in map[i:i+17]])
            fmaps.append(np.transpose(fmap))

        return fmaps

    def get_data(self, data,base_dir = '../data',attrs_dir = '../data/protocol_V2'):

        if data == 'train':
            fbank_dir = os.path.join(base_dir, 'ASVspoof2017_V2_train_fbank')
            attrs_dir = os.path.join(attrs_dir, 'ASVspoof2017_V2_train.trn.txt')
        elif data == 'dev':
            fbank_dir = os.path.join(base_dir, 'ASVspoof2017_V2_train_dev')
            attrs_dir = os.path.join(attrs_dir, 'ASVspoof2017_V2_dev.trl.txt')
        elif data == 'eval':
            fbank_dir = os.path.join(base_dir, 'ASVspoof2017_V2_train_eval')
            attrs_dir = os.path.join(attrs_dir, 'ASVspoof2017_V2_eval.trl.txt')

        #Find Filenames
        filenames = [x for x in os.listdir(fbank_dir) if x.endswith(".cmp")]
        #Guess Corresponding Wav Filenames (for later use)
        self.wavfilenames = [x[:-4]+'.wav' for x in filenames]

        print("\nGetting \""+data+"\" attributes...")
        attrs=[]
        with open(attrs_dir) as f :
            content = f.readlines()
            content = [x.strip() for x in content]
            for c in content:
                att = c.split(' ')
                attrs.append(att)
        print("Reading \""+data+"\" CMP files...")
        for filename in progressbar.progressbar(filenames) :
            # print(filename)

            with open(fbank_dir + '/'+filename, 'r') as fid:
                # filterbank = map(float,fid)
                filterbanks = np.fromfile(fid, dtype=np.float32)
                # filterbanks = np.array(filterbanks)

                #reshape the filterbanks to create banks of size 64
                filterbanks = np.reshape(filterbanks, (-1, 64))

                #make fmaps_list
                fmaps_temp = self.make_fmaps(filterbanks)

                #This is why I love python...
                fmaps_attr_temp = [attr[:] for attr in attrs if attr[0] == self.wavfilenames[filenames.index(filename)] for k in fmaps_temp]

                self.fmaps_list.extend(fmaps_temp)
                self.fmaps_attr_list.extend(fmaps_attr_temp)



    # c = list(zip(fmaps_list, fmaps_attr_list))
    # random.shuffle(c)
    # fmaps_list, fmaps_attr_list = zip(*c)

