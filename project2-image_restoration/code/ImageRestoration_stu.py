import numpy as np
from scipy.misc import imread, imresize, imsave
import scipy.stats as stats
import matplotlib.pyplot as plt
import h5py
normpdf = stats.norm.pdf
norm = np.linalg.norm
pinv = np.linalg.pinv
dot = np.dot
t = np.transpose

def im2double(im):
    info = np.iinfo(im.dtype)
    return im.astype(np.double) / info.max

def imwrite(im, filename):
    img = np.copy(im)
    img = img.squeeze()
    if img.dtype==np.double:
        #img = np.array(img*255, dtype=np.uint8)
        img = img * np.iinfo(np.uint8).max
        img = img.astype(np.uint8)
    imsave(filename, img)


datafolder = '../data'
resfolder = '../result'
samples = {'A':0.8, 'B':0.4, 'C':0.6}
for testName, noiseRatio in samples.items():
    Img = im2double(imread('{}/{}_ori.png'.format(datafolder,testName)))
    Img[(Img==0)]=0.01
    # imwrite(Img, '{}/{}_ori.png'.format(datafolder,testName))
    corrImg = im2double(imread('{}/{}.png'.format(datafolder,testName)))
    ############### generate corrupted image ###############

    if len(Img.shape) == 2:
        Img = Img[:, :, np.newaxis]
        corrImg = corrImg[:, :, np.newaxis]
    rows, cols, channels = Img.shape

    ## generate noiseMask and corrImg
    '''
    noiseMask = np.ones((rows, cols, channels))
    subNoiseNum = round(noiseRatio * cols)
    for k in range(channels):
        for i in range(rows):
            tmp = np.random.permutation(cols)
            noiseIdx = np.array(tmp[:subNoiseNum])
            noiseMask[i, noiseIdx, k] = 0
    corrImg = Img * noiseMask
    imwrite(corrImg, '{}/{}.png'.format(datafolder,testName))
    '''

    noiseMask = np.array(corrImg!=0, dtype='double')

    ## standardize the corrupted image
    minX = np.min(corrImg)
    maxX = np.max(corrImg)
    corrImg = (corrImg - minX)/(maxX-minX)

    ## ======================learn the coefficents in regression function======================
    # In this section, we use gaussian kernels as the basis functions. And we
    # do regression analysis each row at a time,this is just a baseline method.
    # You can analysis by block or use other methods.Complete the image restoration algorithm.
    basisNum = 50 # number of basis functions
    sigma = 0.01 # standard deviation
    Phi_mu = np.linspace(1, cols, basisNum)/cols # mean value of each basis function
    Phi_sigma = sigma * np.ones((basisNum)) # set the standard deviation to the same value for brevity















    ## show the restored image
    if Img.shape[2] == 1:
        Img = Img.squeeze()
        corrImg = corrImg.squeeze()
        resImg = resImg.squeeze()
        plt.subplot(1, 3, 1)
        plt.imshow(Img, cmap='gray')
        plt.subplot(1, 3, 2)
        plt.imshow(corrImg, cmap='gray')
        plt.subplot(1, 3, 3)
        plt.imshow(resImg, cmap='gray')
        plt.show()
    else:
        plt.subplot(1, 3, 1)
        plt.imshow(Img)
        plt.subplot(1, 3, 2)
        plt.imshow(corrImg)
        plt.subplot(1, 3, 3)
        plt.imshow(resImg)
        plt.show()

    prefix = '%s/%s_%.1f_%d'%(resfolder, testName, noiseRatio, basisNum)
    fileName = prefix+'_result.h5'
    h5f = h5py.File(fileName,'w')
    h5f.create_dataset("basisNum", dtype='uint8', data=basisNum)
    h5f.create_dataset("sigma", dtype='double', data=sigma)
    h5f.create_dataset("Phi_mu", dtype='double', data=Phi_mu)
    h5f.create_dataset("Phi_sigma", dtype='double', data=Phi_sigma)
    h5f.create_dataset("resImg", dtype='double', data=resImg)
    h5f.create_dataset("noiseMask", dtype='double', data=noiseMask)
    h5f.close()
    #h5f = h5py.File(fileName,'r')
    #resImg = h5f.get('resImg').value
    #noiseMask = h5f.get('noiseMask').value
    #...
    #h5f.close()

    ## compute error
    im1 = Img.flatten()
    im2 = corrImg.flatten()
    im3 = resImg.flatten()
    print((
        '{}({}):\n'
        'Distance between original and corrupted: {}\n'
        'Distance between original and reconstructed (regression): {}'
    ).format(testName, noiseRatio, norm(im1-im2, 2), norm(im1-im3, 2)))

    ## store figure
    imwrite(resImg, '{}.png'.format(prefix))