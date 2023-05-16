import os
import time
import cv2

from skimage.metrics import structural_similarity as ssim
from matrix_completion_algorithms import *

img_folder = 'data'
mse_scores_list = []
psnr_scores_list = []
ssim_scores_list = []

fista_times_list = []
admm_times_list = []
svt_times_list = []

iter = 1
for image in os.listdir(img_folder):
    print("Processing image number ", iter)

    # get path to image
    image_path = img_folder + '/' + image
    
    # read in the original image
    original_image = cv2.imread(image_path, 0)
    # normalize the image
    original_image = original_image / 255
    
    # select noise percentage to infuse the original with
    noise_percentage=0.5
    # add random noise to original image
    mask = np.random.choice(a=[0,1], size=original_image.shape, p=[noise_percentage, 1-noise_percentage])
    noise_image = np.multiply(original_image, mask)
    
    # locate observed entries from the original image
    observed_entries = np.where(mask != 0)
    
    # run fista with nesterov on image
    start_time = time.time()
    recs_fista = fista_with_nesterov(noise_image, observed_entries)
    print("fista ", time.time() - start_time)
    fista_times_list.append(time.time() - start_time)
    
    # run admm on image    
    start_time = time.time()
    recs_admm = admm(noise_image, observed_entries)
    print("admm ", time.time() - start_time)
    admm_times_list.append(time.time() - start_time)

    # run SVT on image
    start_time = time.time()
    recs_svt = svt(M=noise_image, eps=1e-4, delta=9e-1, k0=9.23e-1, l=1, steps=1000)
    print("svt ", time.time() - start_time)
    svt_times_list.append(time.time() - start_time)
    
    # evaluate metrics
    mse_images = []
    psnr_images = []
    ssim_images = []

    for rec_image in [recs_fista, recs_admm, recs_svt]:
        
        # calculate MSE
        mse = np.mean((original_image - rec_image) ** 2)
        mse_images.append(mse)
        
        # Calculate the PSNR value
        max_pixel_value = np.max(original_image)
        psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
        psnr_images.append(psnr)
        
        # Calculate the SSIM value
        score = ssim(original_image, rec_image, data_range=1)
        ssim_images.append(score)
        
    mse_scores_list.append(mse_images)
    psnr_scores_list.append(psnr_images)
    ssim_scores_list.append(ssim_images)
    iter += 1
    
mse_scores_list = np.array(mse_scores_list)
psnr_scores_list = np.array(psnr_scores_list)
ssim_scores_list = np.array(ssim_scores_list)

print("Average Scores: ")
print("Fista MSE: ", np.mean(mse_scores_list[:,0]))
print("ADMM MSE: ", np.mean(mse_scores_list[:,1]))
print("SVT MSE: ", np.mean(mse_scores_list[:,2]))
print()
print("Fista PSNR: ", np.mean(psnr_scores_list[:,0]))
print("ADMM PSNR: ", np.mean(psnr_scores_list[:,1]))
print("SVT PSNR: ", np.mean(psnr_scores_list[:,2]))
print()
print("Fista SSIM: ", np.mean(ssim_scores_list[:,0]))
print("ADMM SSIM: ", np.mean(ssim_scores_list[:,1]))
print("SVT SSIM: ", np.mean(ssim_scores_list[:,2]))

print("Average Times: ")
print("Fista time: ", np.mean(fista_times_list))
print("ADMM time: ", np.mean(admm_times_list))
print("SVT time: ", np.mean(svt_times_list))