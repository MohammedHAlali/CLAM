import os
import glob

in_path = '/work/deogun/alali/data/lung_png/train/cancer/'
len_list = []
for name in os.listdir(in_path):
    name_path = os.path.join(in_path, name)
    filenames = glob.glob(os.path.join(name_path, '*.png'))
    print('In name {}, there are {} tiles'.format(name, len(filenames)))
    len_list.append(len(filenames))


print('min: ', min(len_list))
print('max: ', max(len_list))
print('avg: ', (sum(len_list)/len(len_list)))
