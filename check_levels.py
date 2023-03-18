'''
Here, we read each WSI, if 4 levels available, then it's 40x. if 3 levels available, then it's 20x.
We need this because 20x WSIs need to be processed differently when calling create_patches_fpy.py
we need to give patch_level 1 for 40x files
but patch_level 0 for 20x files
'''

import argparse
import os
import openslide

parser = argparse.ArgumentParser(description='check WSI levels')
parser.add_argument('--phase', type=str, default='train')
parser.add_argument('--group', type=str, default=0)
parser.add_argument('--classname', type=str, default='normal')
args = parser.parse_args()
print('arguments: ', args)

wsi_path = '/common/deogun/alali/data/lung_svs/{}/{}/{}'.format(args.phase, args.classname, args.group)
print('reading wsi files from: ', wsi_path)
for f in os.listdir(wsi_path):
    print('filename: ', f)
    file_path = os.path.join(wsi_path, f)
    #print('trying to open file: ', file_path)
    im = openslide.OpenSlide(file_path)
    print('number of levels: ', len(im.level_dimensions))
    print('level dimensions: ', im.level_dimensions)
    #if(len(im.level_dimensions) == 4):
    #    raise ValueError('ERROR: found 4 level')

