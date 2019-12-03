
import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def rescale_to_255(image):
    return ((image - image.min()) * (1/(image.max() - image.min()) * 255)).astype('uint8')

def read_process_img(img_path):
    img = plt.imread(img_path).astype("float64")
    if img.ndim > 2:
        img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    img = rescale_to_255(img)

    # try different hyper parameters for face detection and combine the best results.

    # faces = face_cascade.detectMultiScale(img, 1.3, 5)
    faces = face_cascade.detectMultiScale(img, 1.1, 3)
    # faces = face_cascade.detectMultiScale(img, 1.3, 3)
    for (x, y, w, h) in faces:
        img = img[y:y + h, x:x + w]
        break
    try:
        # Topic: Subsampling with Gaussian pre-ﬁltering

        img = cv2.GaussianBlur(img, (9, 9), 0)
        img = cv2.resize(img, (48, 48))
    except:
        print("err")
    return img

txt_filenames = glob.glob("txt_labels" + '/*.'+"txt")

print(txt_filenames)

if not os.path.exists('./CK+'):
    os.makedirs('./CK+')
    os.makedirs('./CK+/1')
    os.makedirs('./CK+/2')
    os.makedirs('./CK+/3')
    os.makedirs('./CK+/4')
    os.makedirs('./CK+/5')
    os.makedirs('./CK+/6')
    os.makedirs('./CK+/7')


total = 0
for filename in txt_filenames:
    f = open(filename,"r")
    content = f.readline()
    label = int(content.split(".")[0].replace(" ",""))
    # print(label)
    pure_img_filename = filename[11:28]
    the_number = pure_img_filename[-2:]
    the_number2 = str( int(the_number) - 1 )
    if len(the_number2) == 1:
        the_number2 = '0' + the_number2
    pure_img_filename2 = pure_img_filename[:-2] + the_number2
    the_number3 = str( int(the_number) - 2 )
    if len(the_number3) == 1:
        the_number3 = '0' + the_number3
    pure_img_filename3 = pure_img_filename[:-2] + the_number3

    target_folder = 'CK+//' + str(label) + '//'

    print(pure_img_filename)

    full_img_path1 = 'origin_imgs//'+ pure_img_filename + '.png'
    preprocessed_img1 = read_process_img(full_img_path1)
    full_img_path1 = full_img_path1.replace('origin_imgs//',target_folder)
    Image.fromarray(preprocessed_img1, 'L').save(full_img_path1)

    print(pure_img_filename2)

    full_img_path2 = 'origin_imgs//'+ pure_img_filename2 + '.png'
    preprocessed_img2 = read_process_img(full_img_path2)
    full_img_path2 = full_img_path2.replace('origin_imgs//',target_folder)
    Image.fromarray(preprocessed_img2, 'L').save(full_img_path2)

    print(pure_img_filename3)

    full_img_path3 = 'origin_imgs//'+ pure_img_filename3 + '.png'
    preprocessed_img3 = read_process_img(full_img_path3)
    full_img_path3 = full_img_path3.replace('origin_imgs//',target_folder)
    Image.fromarray(preprocessed_img3, 'L').save(full_img_path3)



    total +=1
print(total)

