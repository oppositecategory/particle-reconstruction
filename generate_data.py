import numpy as np
import matplotlib.pyplot as plt 
import cv2

import mrcfile

from PIL import Image 

def translate_image(image, tx, ty):
    # Continous translation using bilinear interpolation
    M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32) 
    translated = cv2.warpAffine(image, M, image.shape, flags=cv2.INTER_LINEAR)  
    return translated   


def generate_data(N, SNR=0.5, size=256):
    img = Image.open("image.png").convert("L").resize((256,256))
    img = np.array(img)

    n = size
    thetas = np.random.uniform(0,2*np.pi,N)
    translations = np.random.multivariate_normal(mean=[0,0], 
                                             cov=(0.05*n)**2 * np.identity(2),
                                             size=N)
    scales = np.random.beta(a=2,b=5,size=N)*5
    noise_matrices = np.random.normal(scale=np.var(img)/SNR,size=(N,n,n))

    center =  (n//2,n//2)
    rotated_images = np.array(
        [
            cv2.warpAffine(img,
                           cv2.getRotationMatrix2D(center,angle * 180 / np.pi,scale=1),dsize=(n,n)) 
            for angle in thetas
    ])

    translated_images = np.array([
        translate_image(img, t[0],t[1]) for img,t in zip(rotated_images,translations)
    ])
    scaled_images = np.array([s*img for img,s in zip(translated_images,scales)])
    data = scaled_images + noise_matrices

    data = data.astype(np.float32)

    with mrcfile.new('data.mrc',overwrite=True) as mrc:
        mrc.set_data(data)

if __name__ == "__main__":
    generate_data(10**5)