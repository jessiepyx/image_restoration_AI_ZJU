import numpy as np
from scipy.misc import imread, imresize, imsave
import scipy.stats as stats
import matplotlib.pyplot as plt
import h5py
from sklearn.neighbors import KDTree

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


########## options #############
TEST = True
MAKE_NOISE = False
KNN = True

if TEST:
    HAVE_ORIGINAL_IMAGE = False
else:
    HAVE_ORIGINAL_IMAGE = True

if not HAVE_ORIGINAL_IMAGE:
    MAKE_NOISE = False
################################


datafolder = '../data'
resfolder = '../result'
if TEST:
    samples = {'A': 0.8, 'B': 0.4, 'C': 0.6}
else:
    samples = {'castle': 0.5, 'car': 0.7}
for testName, noiseRatio in samples.items():
    if HAVE_ORIGINAL_IMAGE:
        Img = im2double(imread('{}/{}_ori.png'.format(datafolder, testName)))
        Img[(Img == 0)] = 0.01
        # imwrite(Img, '{}/{}_ori.png'.format(datafolder, testName))
    if not MAKE_NOISE:
        corrImg = im2double(imread('{}/{}.png'.format(datafolder, testName)))

    ############### generate corrupted image ###############

    if HAVE_ORIGINAL_IMAGE:
        if len(Img.shape) == 2:
            Img = Img[:, :, np.newaxis]
        rows, cols, channels = Img.shape

    ## generate noiseMask and corrImg
    if MAKE_NOISE:
        noiseMask = np.ones((rows, cols, channels))
        subNoiseNum = round(noiseRatio * cols)
        for k in range(channels):
            for i in range(rows):
                tmp = np.random.permutation(cols)
                noiseIdx = np.array(tmp[:subNoiseNum])
                noiseMask[i, noiseIdx, k] = 0
        corrImg = Img * noiseMask
        imwrite(corrImg, '{}/{}.png'.format(datafolder, testName))

    if len(corrImg.shape) == 2:
        corrImg = corrImg[:, :, np.newaxis]
    rows, cols, channels = corrImg.shape

    noiseMask = np.array(corrImg != 0, dtype='double')

    ## standardize the corrupted image
    minX = np.min(corrImg)
    maxX = np.max(corrImg)
    corrImg = (corrImg - minX)/(maxX-minX)

    ## ======================learn the coefficents in regression function======================
    # In this section, we use gaussian kernels as the basis functions. And we
    # do regression analysis each row at a time, this is just a baseline method.
    # You can analysis by block or use other methods.Complete the image restoration algorithm.
    basisNum = 50  # number of basis functions
    sigma = 0.01  # standard deviation
    Phi_mu = np.linspace(1, cols, basisNum)/cols  # mean value of each basis function
    Phi_sigma = sigma * np.ones(basisNum)  # set the standard deviation to the same value for brevity

    h = corrImg.shape[0]
    w = corrImg.shape[1]
    resImg = corrImg.copy()

    if KNN:  # k-Nearest Neighbor
        # for each channel
        for c in range(corrImg.shape[2]):
            inputImg = corrImg[:, :, c]
            mask = noiseMask[:, :, c]

            # prepare data
            train_x = []
            test_x = []
            for i, tmp in enumerate(mask):
                for j, val in enumerate(tmp):
                    if val == 1:
                        train_x.append([i, j])
                    else:
                        test_x.append([i, j])
            train_x = np.array(train_x)
            test_x = np.array(test_x)
            train_y = [inputImg[ind[0]][ind[1]] for ind in train_x]
            train_y = np.array(train_y)

            # build KD-Tree
            kd_tree = KDTree(train_x)
            dist, ind = kd_tree.query(test_x, k=3)

            # predict
            pred_y = np.sum(train_y[ind] / dist, axis=1) / np.sum(1 / dist, axis=1)

            # restore image
            for i in range(test_x.shape[0]):
                resImg[test_x[i][0]][test_x[i][1]][c] = pred_y[i]

    else:  # default: regression analysis by line with Gaussian kernel
        # for each channel
        for c in range(corrImg.shape[2]):
            inputImg = corrImg[:, :, c]
            mask = noiseMask[:, :, c]

            # analysis by line
            for i in range(h):
                line = inputImg[i]
                tp = np.array([line[j] for j in range(w) if mask[i][j] == 1])
                xp = np.array([np.linspace(0, 1, num=w)[j] for j in range(w) if mask[i][j] == 1])
                xn = np.array([np.linspace(0, 1, num=w)[j] for j in range(w) if mask[i][j] == 0])

                # Gaussian basis function
                phi_p = np.zeros((basisNum, xp.shape[0]))
                phi_n = np.zeros((basisNum, xn.shape[0]))
                for j in range(basisNum):
                    phi_p[j] = np.exp(-(xp - Phi_mu[j])**2 / (Phi_sigma[j]**2 * 2))
                for j in range(basisNum):
                    phi_n[j] = np.exp(-(xn - Phi_mu[j])**2 / (Phi_sigma[j]**2 * 2))

                W_ml = np.linalg.pinv(phi_p.dot(np.transpose(phi_p))).dot(phi_p).dot(tp)
                tn = np.transpose(W_ml).dot(phi_n)

                # restore image
                cnt = 0
                for j in range(w):
                    if line[j] == 0:
                        resImg[i][j][c] = tn[cnt]
                        cnt = cnt + 1

    # clip to [0, 1]
    resImg[resImg > 1.0] = 1.0
    resImg[resImg < 0.0] = 0.0

    ## show the restored image
    if corrImg.shape[2] == 1:
        if HAVE_ORIGINAL_IMAGE:
            Img = Img.squeeze()
            plt.subplot(1, 3, 1)
            plt.imshow(Img, cmap='gray')
        corrImg = corrImg.squeeze()
        resImg = resImg.squeeze()
        plt.subplot(1, 3, 2)
        plt.imshow(corrImg, cmap='gray')
        plt.subplot(1, 3, 3)
        plt.imshow(resImg, cmap='gray')
        plt.show()
    else:
        if HAVE_ORIGINAL_IMAGE:
            plt.subplot(1, 3, 1)
            plt.imshow(Img)
        plt.subplot(1, 3, 2)
        plt.imshow(corrImg)
        plt.subplot(1, 3, 3)
        plt.imshow(resImg)
        plt.show()

    prefix = '%s/%s_%.1f_%d' % (resfolder, testName, noiseRatio, basisNum)
    if KNN:
        prefix = prefix + 'kNN'
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
    if HAVE_ORIGINAL_IMAGE:
        im1 = Img.flatten()
        im2 = corrImg.flatten()
        im3 = resImg.flatten()
        print((
            '{}({}):\n'
            'Distance between original and corrupted: {}\n'
            'Distance between original and reconstructed (regression): {}'
        ).format(testName, noiseRatio, np.linalg.norm(im1-im2, 2), np.linalg.norm(im1-im3, 2)))

    ## store figure
    imwrite(resImg, '{}.png'.format(prefix))
