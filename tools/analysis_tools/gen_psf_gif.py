import argparse
import os
import cv2 as cv    
# from PIL import Image
from mediapy import write_video
def read_psf(log_path):
    imgs_path = os.path.join(log_path, 'visualizations')
    imgs = os.listdir(imgs_path)
    psf_list = []
    for img in imgs:
        if img.endswith('psf.png'):
            psf_list.append(img)
    #sort psf with order
    psf_list.sort(key=lambda x:int(x.split('_')[0]))
    return psf_list
#generate gif with a list of images
def gen_gif_from_list(log_path, psf_list):
    frames = []
    for image_file in psf_list:
        image_path = os.path.join(log_path,"visualizations", image_file)
        img = cv.imread(image_path)
        frames.append(img)
    # Save the frames as a GIF
    gif_path = os.path.join(log_path, 'psf.gif')
    write_video(gif_path, frames[::5], fps=25, codec="gif")
    # imageio.mimsave(gif_path, frames, format='GIF', duration=0.5) # Adjust the duration as needed
    print(f"Successfully created the GIF: {gif_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='gen psf gif')
    parser.add_argument('log_path', help='log path')
    args = parser.parse_args()
    psf_list = read_psf(args.log_path)
    gen_gif_from_list(args.log_path, psf_list)