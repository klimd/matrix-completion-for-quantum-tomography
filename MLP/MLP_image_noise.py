import cv2
import os

import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def train_model(trainX, testX, noise_percentage=0.2, learning_rate=0.001, num_layers=1, epochs=10, batch_size=1, num_of_test=1):
    clear_session()
    # Normalize
    trainX = trainX.astype('float32') / 255
    testX = testX.astype('float32') / 255
        
    # Make 1D image
    pixel_size = trainX.shape[1] * trainX.shape[2]
    trainX = trainX.reshape(trainX.shape[0], pixel_size)
    testX = testX.reshape(testX.shape[0], pixel_size)
    # Add noise
    trainX_noise = trainX + noise_percentage * np.random.normal(loc=0.0, scale=1.0, size=trainX.shape)
    testX_noise = testX + noise_percentage * np.random.normal(loc=0.0, scale=1.0, size=testX.shape)
    trainX_noise = np.clip(trainX_noise, 0., 1.)
    testX_noise = np.clip(testX_noise, 0., 1.)
    
    # Make model
    model = Sequential()
    if num_layers == 1:
        model.add(Dense(10, input_dim=pixel_size, activation='relu'))
        model.add(Dense(pixel_size, activation='sigmoid'))
        
    elif num_layers == 2:
        model.add(Dense(10, input_dim=pixel_size, activation='sigmoid'))
        model.add(Dense(20, activation='sigmoid'))
        model.add(Dense(pixel_size, activation='sigmoid'))
    else:
        model.add(Dense(10, input_dim=pixel_size, activation='sigmoid'))
        for _ in range(num_layers-1):
            model.add(Dense(20, activation='sigmoid'))
        model.add(Dense(pixel_size, activation='sigmoid'))
    
    opt = Adam(learning_rate=0.01)
    model.compile(loss='mse', optimizer=opt)
    model.fit(trainX_noise, trainX, validation_data=(testX_noise, testX), epochs=epochs, batch_size=batch_size)
    
    prediction = model.predict(testX_noise)
    
    prediction = np.reshape(prediction, (num_of_test, size_x, size_y))
    testX_noise = np.reshape(testX_noise, (-1, size_x, size_y))
    return prediction, testX_noise


# Load all images in the folder
img_folder = 'data'
original_images = []
for image in os.listdir(img_folder):
    original_images.append(cv2.imread(img_folder + "/" + image, 0))
    
# Resize them so that they are all equal size
resized = []
size_x = 120
size_y = 120
for image in original_images:
    resized.append(cv2.resize(image, [size_x, size_y], interpolation = cv2.INTER_AREA))
    
resized = np.array(resized)
num_of_test = int(len(original_images) * 0.2)
testX = resized[0:num_of_test] 
trainX = resized[num_of_test:-1]

print(len(original_images), resized.shape, num_of_test, trainX.shape, testX.shape)

pred, testX_noise = train_model(trainX, testX, noise_percentage=0.5, learning_rate=1e-2, num_layers=100, epochs=100, batch_size=32, num_of_test=num_of_test)


# evaluate metrics
mse_scores_list = []
psnr_scores_list = []
ssim_scores_list = []

for i in range(len(pred)):
    
    rec_image = pred[i]
    original_image = testX[i]
    
     # calculate MSE
    mse = np.mean((original_image - rec_image) ** 2)
    mse_scores_list.append(mse)
        
    # Calculate the PSNR value
    max_pixel_value = np.max(original_image)
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    psnr_scores_list.append(psnr)
        
    # Calculate the SSIM value
    score = ssim(original_image, rec_image, data_range=1)
    ssim_scores_list.append(score)
    
mse_scores_list = np.array(mse_scores_list)
psnr_scores_list = np.array(psnr_scores_list)
ssim_scores_list = np.array(ssim_scores_list)

print("Average Scores: ")
print("MLP MSE: ", np.mean(mse_scores_list))
print()
print("MLP PSNR: ", np.mean(psnr_scores_list))
print()
print("MLP SSIM: ", np.mean(ssim_scores_list))