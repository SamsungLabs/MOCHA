import pandas as pd
import shutil
import os
from PIL import Image

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

data_root = '../OpenImagesRaw'

remapping_classes = {
                     'Bottle': 'bottle',
                     'Soap dispenser': 'bottle',
                     'Personal care': 'bottle',
                     'Fruit': 'fruit',
                     'Glasses': 'glasses',
                     'Sunglasses': 'glasses',
                     'Mug': 'mug',
                     'Coffee cup': 'mug',
                     'Pen': 'pen',
                     'Snack': 'snack',
                     'Power plugs and sockets': 'plug',
                     'Mobile phone': 'phone',
                     'Scissors': 'scissors',
                     'Light bulb': 'light',
                     'Tin can': 'can',
                     'Football': 'ball',
                     'Cricket ball': 'ball',
                     'Volleyball (Ball)': 'ball',
                     'Golf ball': 'ball',
                     'Tennis ball': 'ball',
                     'Rugby ball': 'ball',
                     'Remote control': 'remote',
                     'Backpack': 'backpack',
                     'Teddy bear': 'toy',
                     'Toy': 'toy',
                     'Doll': 'toy',
                     'House': 'house',
                     'Tree house': 'house',
                     'Cat': 'cat',
                     'Chair': 'chair',
                     'Alarm clock': 'clock',
                     'Clock': 'clock',
                     'Digital clock': 'clock',
                     'Wall clock': 'clock',
                     'Watch': 'clock',
                     'Footwear': 'boot',
                     'Boot': 'boot',
                     'Teapot': 'teapot',
                     'Dog': 'dog',
                     'Candle': 'decoration',
                     'Vase': 'decoration',
                     'Flowerpot': 'decoration',
                     'Chime': 'decoration',
                     'Table': 'table',
                     'Coffee table': 'table',
                     'Kitchen & dining room table': 'table',
                     'Book': 'book',
                     'Ring binder': 'book',
                     'Wine glass': 'glass',
                     'Drink': 'glass',
                     'Luggage and bags': 'bags',
                     'Computer mouse': 'mouse',
                     'Glove': 'glove',
                     'Pencil case': 'case',
                     'Hair spray': 'spray',
                     'Cooking spray': 'spray'
                    }

classes_to_ids = {
                  'bottle': 0,
                  'fruit': 1,
                  'glasses': 2,
                  'mug': 3,
                  'pen': 4,
                  'snack': 5,
                  'plug': 6,
                  'phone': 7,
                  'scissors': 8,
                  'light': 9,
                  'can': 10,
                  'ball': 11,
                  'remote': 12,
                  'backpack': 13,
                  'toy': 14,
                  'house': 15,
                  'cat': 16,
                  'chair': 17,
                  'clock': 18,
                  'boot': 19,
                  'teapot': 20,
                  'dog': 21,
                  'decoration': 22,
                  'table': 23,
                  'book': 24,
                  'glass': 25,
                  'bags': 26,
                  'mouse': 27,
                  'glove': 28,
                  'case': 29,
                  'spray': 30
                  }


description = pd.read_csv(os.path.join(data_root, 'oidv6-class-descriptions.csv'))

csv_file_map={'val': 'validation', 'test': 'test', 'train': 'oidv6-train'}
folder_to_folder_map={'val': 'validation', 'test': 'test', 'train': 'train'}

for t in csv_file_map.keys():
    print(f'\n Running {t} split \n')

    df = pd.read_csv(os.path.join(data_root, f'{csv_file_map[t]}-annotations-bbox.csv'))

    # Samples to exclude:
    # - IsDepiction: 1
    df.drop(df[df['IsDepiction']==1].index, inplace=True)

    # For each class, move to data folder the classes we want to keep.
    for src_class_name, dst_folder in remapping_classes.items():
        print(src_class_name)
        src_class_id = description.loc[description['DisplayName']==src_class_name]['LabelName']
        assert len(src_class_id) == 1, f"Class [{src_class_name}] not found"
        src_class_id = src_class_id.values[0]  # string source class identifier

        new_df = df.drop(df[df['LabelName']!=src_class_id].index)
        new_df = new_df.reset_index()

        for _, sample in new_df.iterrows():

            # Object Detection dataset
            imagename = sample['ImageID']
            # 1) Copy image to destination folder
            save_dir=os.path.join('object_detection', t)
            makedirs(save_dir)
            shutil.copyfile(os.path.join(data_root, folder_to_folder_map[t], f'{imagename}.jpg'),
                            os.path.join(save_dir, f'{imagename}.jpg'))

            # 2) Extract bounding boxes in the required format
            orig_im = Image.open(os.path.join(data_root, folder_to_folder_map[t], f'{imagename}.jpg'))
            x_min, x_max, y_min, y_max = sample.XMin, sample.XMax, sample.YMin, sample.YMax
            true_label = remapping_classes[src_class_name]

            # 3) convert to proper format
            width = (x_max-x_min)
            height = (y_max-y_min)
            center = (x_min+width/2, y_min+height/2) #not integer division as they are fractions

            # 4) save
            save_dir=os.path.join('object_detection', 'labels', t)
            makedirs(save_dir)
            s = str(classes_to_ids[true_label]) + " " + str(center[0]) + " " + str(center[1]) + " " + str(width) + " " + str(height) + "\n"
            with open(os.path.join(save_dir, f'{imagename}.txt'), "a+") as file:
                file.writelines(s)


            # Image Recognition dataset
            # 1) Crop image
            cropped_im = None
            true_bbs = [x_min*orig_im.size[0], x_max*orig_im.size[0], y_min*orig_im.size[1], y_max*orig_im.size[1]]

            offset=10
            max_size = max(true_bbs[1]-true_bbs[0]+2*offset, true_bbs[3]-true_bbs[2]+2*offset)
            center = ( true_bbs[0]+(true_bbs[1]-true_bbs[0])//2, true_bbs[2]+(true_bbs[3]-true_bbs[2])//2 )

            # left, right = max(0, true_bbs[0] - offset), min(true_bbs[1] + offset, orig_im.size[0])  # crop according to bbs only
            left, right = max(0, center[0]-max_size//2), min(center[0]+max_size//2, orig_im.size[0])
            up, down = max(0, center[1]-max_size//2), min(center[1]+max_size//2, orig_im.size[1])
            cropped_im = orig_im.crop((left, up, right, down))

            # 2) Copy cropped image to destination folder (class-wise)
            save_dir = os.path.join('image_recognition', t, true_label)
            makedirs(save_dir)
            cropped_im.save(os.path.join(save_dir, f"{imagename}_{sample['index']}.jpg"))
