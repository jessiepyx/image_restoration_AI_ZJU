# image_restoration_AI_ZJU
Zhejiang University Course: Artificial Intelligence (19 Spring) Individual Project -- Image Restoration.  
Course website is [here](http://10.15.82.131/cv/course/view.php?id=31).

This project uses Python 3.6.
## Discription
The task is to build regression models to reconstruct corrupted images. Input images are masked by random noise with specified noise rate. 

I implemented two regression models:
- Linear Model with Gassian Basis Function (and analysis by line, as specified in assignment instruction)
- k-Nearest Neighbor Model (which performs better)
## Results
### 0.8 noise (Corrupted/Gaussian Basis/kNN)
![A](https://github.com/jessiepyx/image_restoration_AI_ZJU/blob/master/project2-image_restoration/data/A.png "Corrupted")
![A_gauss](https://github.com/jessiepyx/image_restoration_AI_ZJU/blob/master/project2-image_restoration/result/A_0.8_50.png "Gaussian Basis")
![A kNN](https://github.com/jessiepyx/image_restoration_AI_ZJU/blob/master/project2-image_restoration/result/A_0.8_50kNN.png "kNN")
### 0.4 noise (Corrupted/Gaussian Basis/kNN)
![B](https://github.com/jessiepyx/image_restoration_AI_ZJU/blob/master/project2-image_restoration/data/B.png "Corrupted")
![B_gauss](https://github.com/jessiepyx/image_restoration_AI_ZJU/blob/master/project2-image_restoration/result/B_0.4_50.png "Gaussian Basis")
![B kNN](https://github.com/jessiepyx/image_restoration_AI_ZJU/blob/master/project2-image_restoration/result/B_0.4_50kNN.png "kNN")
### 0.6 noise (Corrupted/Gaussian Basis/kNN)
![C](https://github.com/jessiepyx/image_restoration_AI_ZJU/blob/master/project2-image_restoration/data/C.png "Corrupted")
![C_gauss](https://github.com/jessiepyx/image_restoration_AI_ZJU/blob/master/project2-image_restoration/result/C_0.6_50.png "Gaussian Basis")
![C kNN](https://github.com/jessiepyx/image_restoration_AI_ZJU/blob/master/project2-image_restoration/result/C_0.6_50kNN.png "kNN")
### 0.7 noise (Original/Corrupted/Gaussian Basis/kNN)
![car_ori](https://github.com/jessiepyx/image_restoration_AI_ZJU/blob/master/project2-image_restoration/data/car_ori.png "Original")
![car](https://github.com/jessiepyx/image_restoration_AI_ZJU/blob/master/project2-image_restoration/data/car.png "Corrupted")
![car_gauss](https://github.com/jessiepyx/image_restoration_AI_ZJU/blob/master/project2-image_restoration/result/car_0.7_50.png "Gaussian Basis")
![car kNN](https://github.com/jessiepyx/image_restoration_AI_ZJU/blob/master/project2-image_restoration/result/car_0.7_50kNN.png "kNN")
### 0.5 noise (Original/Corrupted/Gaussian Basis/kNN)
![car_ori](https://github.com/jessiepyx/image_restoration_AI_ZJU/blob/master/project2-image_restoration/data/castle_ori.png "Original")
![car](https://github.com/jessiepyx/image_restoration_AI_ZJU/blob/master/project2-image_restoration/data/castle.png "Corrupted")
![car_gauss](https://github.com/jessiepyx/image_restoration_AI_ZJU/blob/master/project2-image_restoration/result/castle_0.5_50.png "Gaussian Basis")
![car kNN](https://github.com/jessiepyx/image_restoration_AI_ZJU/blob/master/project2-image_restoration/result/castle_0.5_50kNN.png "kNN")
