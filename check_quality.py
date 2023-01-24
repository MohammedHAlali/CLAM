'''
Here, we use the coordinates created by CLAM to extract tiles from the original svs file.
The tiles are then saved in png format.
'''
import openslide
import os
import glob
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--c', type=str, default='adc') #adc/scc/normal
parser.add_argument('--phase', type=str, default='train') #train/val/test
parser.add_argument('--tile_size', type=int, default=1024)
parser.add_argument('--group', type=int, default=1) #1/2/3/4
args = parser.parse_args()
phase = args.phase
c = args.c
group = args.group
print('phase = ', phase, ' class= ', c)
data_path = '/work/deogun/alali/data'
out_path = os.path.join(data_path, 'lung_png', phase, c)
if(not os.path.exists(out_path)):
    os.mkdir(out_path)
    print('directory created: ', out_path)

in_path = os.path.join(data_path,'lung_svs/{}/{}/{}'.format(phase, c, args.group))
print('input path: ', in_path)
tile_size = args.tile_size
#f = 'TCGA-BH-A0AZ-11A-02-BSB.c66f1da4-3d99-4d67-b77a-9a1f3b7735e8'
filenames = glob.glob(os.path.join(in_path, '*.svs'))
print('Number of svs files found: ', len(filenames))
#exit()
for i,f_svs in enumerate(filenames):
    print(i,'- path: ', f_svs)
    l_slash = f_svs.rindex('/')
    case_id = f_svs[l_slash+1:-4]
    print(i, ': ==================== case id: ', case_id)
    #f_svs = os.path.join('svs', f+'.svs')
    print('opening file: ', f_svs)
    im = openslide.OpenSlide(f_svs)
    print('slide levels dimensions: ', im.level_dimensions)
    if(len(im.level_dimensions) < 3):
        raise ValueError('ERROR: found low quality slide, discard')
    else:
        print('good quality slide')
    continue
