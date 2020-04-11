import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from read_write_model import read_model
from scipy.spatial.transform import Rotation


IMAGE_DIRECTORY = 'C:/Datasets/deeparc/teabottle_green/model/undistroted_custom_matching/images'
COLMAP_SPARSE_DIRECTORY = 'C:/Datasets/deeparc/teabottle_green/model/undistroted_custom_matching/sparse/'
SOURCE_IMAGE = ['cam004/cam004_00038.jpg','cam004/cam004_00002.jpg']
REFERNCE_IMAGE = 'cam004/cam004_00000.jpg'
TARGET_IMAGE = 'cam004/cam004_00001.jpg'
OUTPUT_PATH = 'teabottle_half.npz'

def resize_half(image):
    h,w,c = image.shape
    scale = 0.5
    h_new = int(h*scale)
    w_new = int(w*scale)
    print("SIZE", h_new,",", w_new)
    return cv2.resize(image,(w_new,h_new))

def get_image(image_name):
    image_path = os.path.join(IMAGE_DIRECTORY,image_name)
    image = plt.imread(image_path)
    image = image / 255.0
    image = resize_half(image)
    return image

def get_extrinsic(qvec,tvec):
    rotation = Rotation.from_quat([qvec[1],qvec[2],qvec[3],qvec[0]])
    cam_mat = np.eye(4)
    cam_mat[:3,:3] = rotation.as_dcm()
    cam_mat[:3,3] = tvec
    return cam_mat

def get_scale():
    return 0.5

def main():
    os.path.join(IMAGE_DIRECTORY,REFERNCE_IMAGE)
    cameras, images, points3D = read_model(COLMAP_SPARSE_DIRECTORY,'.bin')
    ref_image = [images[i] for i in images if images[i][4] == REFERNCE_IMAGE][0]
    ref_image_pixel = get_image(ref_image[4])
    target_image = [images[i] for i in images if images[i][4] == TARGET_IMAGE][0]
    target_image_pixel = get_image(target_image[4])
    source_images = [images[i] for i in images if images[i][4] in SOURCE_IMAGE]
    source_images_pixel = np.dstack([get_image(s[4]) for s in source_images])
    ref_camera = cameras[ref_image[3]]
    intrinsic = np.ones((3,3))

    scale = get_scale()
    if ref_camera[1] == 'PINHOLE': #COLMAP UNDISTORTED
        fx,fy, cx, cy = ref_camera[4]
        intrinsic[0,0] = fx * scale
        intrinsic[1,1] = fy * scale
        intrinsic[2,0] = cx * scale
        intrinsic[2,1] = cy * scale
    elif ref_camera[1] == 'SIMPLE_RADIAL': #COLMAP DEFAULT
        f, cx, cy, k1 = ref_camera[4]
        intrinsic[0,0] = f * scale
        intrinsic[1,1] = f * scale
        intrinsic[2,0] = cx * scale
        intrinsic[2,1] = cy * scale
        intrinsic[0,1] = k1 # this is aspect ratio resize, so no need to update distortion

    ref_pose = get_extrinsic(ref_image[1],ref_image[2])
    target_pose = get_extrinsic(target_image[1],target_image[2])
    source_poses = np.dstack([get_extrinsic(s[1],s[2]) for s in source_images])
    source_poses = np.moveaxis(source_poses, -1, 0)

    np.savez_compressed(
        OUTPUT_PATH, 
        intrinsics=  np.array([intrinsic]).astype(np.float32),
        src_poses = np.array([source_poses]).astype(np.float32),
        ref_pose =  np.array([ref_pose]).astype(np.float32),
        tgt_pose =  np.array([target_pose]).astype(np.float32),
        src_images =  np.array([source_images_pixel]).astype(np.float32),
        ref_image =  np.array([ref_image_pixel]).astype(np.float32),
        tgt_image = np.array([target_image_pixel]).astype(np.float32)
    )

if __name__ == "__main__":
    main()

"""
image_path = os.path.join(IMAGE_DIRECTORY,REFERNCE_IMAGE)
image = plt.imread(image_path)
image = image / 255.0
image = resize_with_border(image,TARGET_HEIGHT,TARGET_WIDTH)
plt.imsave("output.png",image)
"""