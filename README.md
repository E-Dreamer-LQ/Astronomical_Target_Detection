Detection and Classification of Astronomical Targets with Deep Neural Networks in Wide Field Small Aperture Telescopes
===================================

Introduction
-----------------------------------
Wide field small aperture telescopes are widely used in optical transient observations. Detection and classification of astronomical targets are important steps during data post-processing stage. In this paper, we propose an astronomical targets detection and classification framework based on deep neural networks for images obtained by wide field small aperture telescopes. Our framework adopts the concept of the Faster R-CNN and we further propose to use a modified Resnet-50 as backbone network and a Feature Pyramid Network architecture in our framework. To improve the effectiveness of our framework and reduce requirements of large training set, we propose to use simulated images to train our framework at first and then modify weights of our framework with only a small amount of training data through transfer-learning. We have tested our framework with simulated and real observation data. Comparing with the traditional source detection and classification framework, our framework has better detection ability, particularly for dim astronomical targets. To unleash the transient detection ability of wide field small aperture telescopes, we further propose to install our framework in embedded devices to achieve real-time astronomical targets detection abilities.

Requirements
--------------------------------
* Python 3.5
* Pytorch = 1.1.0
* cuda = 9.0 
* torchvision = 0.3.0
* python3-opencv = 3.3.1
* scipy = 1.4.1 
* imageio = 2.4.1

Results
-------------------------------
The neural network was compared with the sextractor test results in simulated data
.<div align=center><img src="https://github.com/E-Dreamer-LQ/Astronomical_Target_Detection/blob/master/image/simulate_data_result.jpg" /></div>

Training fitting effect of simulated data under 30 epochs
.<div align=center><img src="https://github.com/E-Dreamer-LQ/Astronomical_Target_Detection/blob/master/image/mAP.jpg" width="300"/></div>

The neural network in the 200 images was compared with the recall and precision of the sextractor at each magnitude
.<div align=center><img src="https://github.com/E-Dreamer-LQ/Astronomical_Target_Detection/blob/master/image/nn_vs_sex.jpg" width="300"/></div>

The neural network in the 200 images was compared with the f1 score and f2 score of the sextractor at each magnitude
.<div align=center><img src="https://github.com/E-Dreamer-LQ/Astronomical_Target_Detection/blob/master/image/f1_score_vs.jpg" width="300"/></div>

How to validate
-------------------------------
At first，clone the code:

     git clone https://github.com/E-Dreamer-LQ/Astronomical_Target_Detection.git
    
then you need to complile the project
    
   ```
   cd lib 
   ./make.sh 
   ```
then you need  to complile module of num and roi

  ```
  cd lib/model/AlignPool 
  python3 setup.py install   
  cd lib/model/softnms
  ./compile.sh
  ```
 then check the fits file  which used to validate in the dir of validation/ 

    python3 demo_fits_soft_nms.py --net res50 --dataset star_detect \
            --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
            --cuda --load_dir path/to/model/***.pth --image_dir path/to/validation_images
PS: Some of the real data is in the folder: /validation/xx.fits.

Models 
-------------------------------
You can download the weights to validate the real data at (`password`:pfuc) [Baidu](https://pan.baidu.com/s/1fPvy3zQ9m9vsFvGX2UjZLg)<br />. 

Contributing to the project
-------------------------------
Any pull requests or issues are welcome.
