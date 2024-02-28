import argparse
import os
import cv2 as cv    
# from PIL import Image
from mediapy import write_video
def read_psf(log_path):
    imgs_path = os.path.join(log_path, 'vis_optical_result')
    imgs = os.listdir(imgs_path)
    psf_list = []
    for img in imgs:
        if img.endswith('psf.png'):
            psf_list.append(img)
    #sort psf with order
    psf_list.sort(key=lambda x:int(x.split('_')[1]))
    return psf_list

def read_cap(log_path):
    imgs_path = os.path.join(log_path, 'vis_optical_result')
    imgs = os.listdir(imgs_path)
    cap_list = []
    for img in imgs:
        if img.endswith('after_optical.png'):
            cap_list.append(img)
    #sort psf with order
    cap_list.sort(key=lambda x:int(x.split('_')[1]))
    return cap_list

#generate gif with a list of images
def gen_gif_from_list(log_path, psf_list, name = "psf"):
    frames = []
    for image_file in psf_list:
        image_path = os.path.join(log_path,"vis_optical_result", image_file)
        img = cv.imread(image_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        frames.append(img)
    # Save the frames as a GIF
    # print("frames", len(frames))
    gif_path = os.path.join(log_path, '%s.gif'%name)
    write_video(gif_path, frames[:2000:8], fps=25, codec="gif")
    # imageio.mimsave(gif_path, frames, format='GIF', duration=0.5) # Adjust the duration as needed
    print(f"Successfully created the GIF: {gif_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='gen psf gif')
    parser.add_argument('log_path', help='log path')
    args = parser.parse_args()
    psf_list = read_psf(args.log_path)
    cap_list = read_cap(args.log_path)
    gen_gif_from_list(args.log_path, psf_list)
    gen_gif_from_list(args.log_path, cap_list, name = "cap")