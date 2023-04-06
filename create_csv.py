import os
import csv
base_path = '/work/deogun/alali/CLAM/datasets/'
header = ['case_id', 'slide_id', 'label']
count = 0
with open('datasets/multi_test3.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for phase in ['test3_multi']:
        print('phase: ', phase)
        phase_path = os.path.join(base_path, phase)
        for c in os.listdir(phase_path):
            if('files' in c):
                continue
            class_path = os.path.join(phase_path, c)
            print('class: ', c)
            label = None
            if(c == 'adc'):
                label = 'subtype_1'
            elif(c == 'scc'):
                label = 'subtype_2'
            else:
                label = 'subtype_3'
            print('label: ', label)
            h5_path = os.path.join(class_path, 'h5_files')
            print('h5 path: ', h5_path)
            for imagename in os.listdir(h5_path):
                print('image name: ', imagename)
                patient_id = imagename[:12]
                print('patient id: ', patient_id)
                #dot_index = imagename.index('.')
                slide_id = imagename[:-3]
                print('slide id: ', slide_id)
                writer.writerow([patient_id, slide_id, label])
                count += 1
print('done writing count = ', count)
