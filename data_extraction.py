import numpy as np
import cv2
import os
import shutil

# annotations (MVP dataset)
anno_files_relationships_paths = ['coarse_index_list.txt', 'fine_index_list.txt']
anno_files_relationships_dir = 'dataset/MVP_v1.0/MVP_v1.0/'

# images (VeRi dataset)
img_lookup_paths = ['name_query.txt', 'name_test.txt', 'name_train.txt']
img_lookup_dir = 'dataset/VeRi/'

destination = 'data/'

# how many samples are used for training
test_threshold = 2500

# get all image paths in the VeRi dataset
raw_images = []
for f in img_lookup_paths:
    with open(img_lookup_dir + f) as file:
        d = f.replace('name', 'image').replace('.txt', '/')
        raw_images.extend([img_lookup_dir + d + line.rstrip() for line in file])
   
# see if an annotation is avalable and copy to new destination     
i = 0
all_data = []
for f in anno_files_relationships_paths:
    with open(anno_files_relationships_dir + f) as file:
        for line in file:
            [idx, ds_name, name] = line.split(' ')
            #if ds_name == 'veri': i+= 1 # used to compute how many images do we have
            name = name.replace('\n', '')
            for lookup_dir in img_lookup_paths:
                lookup = img_lookup_dir + lookup_dir.replace('name', 'image').replace('.txt', '/') + name
                if os.path.isfile(lookup):
                    idxx = '0' + idx if 'coarse' in f else idx
                    anno_path = anno_files_relationships_dir + f.split('_')[0] + '_annotation/' + idxx + '.png'
                    all_data.append((lookup, anno_path))
                    
                    train_test = 'train/' if i < test_threshold else 'test/'
                    
                    shutil.copy2(lookup, destination + 'images/' + train_test + str(i) + '.jpg')
                    
                    parsing_anno = cv2.imread(anno_path, cv2.IMREAD_GRAYSCALE)
                    # keep only some annotations (mainly windows), the indices are in the dataset docu
                    if 'coarse' in anno_path:
                        parsing_anno = np.where(np.isin(parsing_anno, [2, 4, 6, 8]), 255, 0)
                    else:
                        parsing_anno = np.where(np.isin(parsing_anno, [28,29,30,31,32,33,34,35,44,45,]), 255, 0)
                    cv2.imwrite(destination + 'annotations/' + train_test + str(i) + '.png', parsing_anno) 
                    
                    i += 1
                    
print('Done!')