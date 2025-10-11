import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):
    filepath_train = filepath + 'train/'
    filepath_val = filepath + 'val/'
    left_fold  = 'image_2/'
    right_fold = 'image_3/'
    disp_L = 'disp_occ_0/'
    dwt_fold = 'dwt_img/'

    image_train = [img for img in os.listdir(filepath_train+left_fold) if img.find('_10') > -1]
    image_val = [img for img in os.listdir(filepath_val+left_fold) if img.find('_10') > -1]
    image_val.sort(key=lambda x: int(x[:-4]))


    left_train   = [filepath_train+left_fold+img for img in image_train]
    right_train  = [filepath_train+right_fold+img for img in image_train]
    dwt_train    = [filepath_train+dwt_fold+img for img in image_train]
    disp_train_L = [filepath_train+disp_L+img for img in image_train]


    left_val   = [filepath_val+left_fold+img for img in image_val]
    right_val  = [filepath_val+right_fold+img for img in image_val]
    dwt_val    = [filepath_val+dwt_fold+img for img in image_val]
    disp_val_L = [filepath_val+disp_L+img for img in image_val]

    return left_train, right_train, dwt_train, disp_train_L, left_val, right_val, dwt_val, disp_val_L