# Trash Detection Using Yolov: ML and CV

## Demo Video
You can find a demo video [here](https://drive.google.com/file/d/17cY-0uWKe79ZZ7QpVN68o5ODuuwlq44U/view?usp=sharing)

## Project Report
You can find a detailed project report [here](https://drive.google.com/file/d/1idQl8_jzbQMLlhJBad9VnitlaxBVqE1R/view?usp=sharing)


## Relevant Yolov Papers and Source Code

* Paper **YOLOv7**: https://arxiv.org/abs/2207.02696

* source code YOLOv7 - Pytorch (use to reproduce results): https://github.com/WongKinYiu/yolov7

----

* Paper **YOLOv4**: https://arxiv.org/abs/2004.10934

* source code YOLOv4 - Darknet (use to reproduce results): https://github.com/AlexeyAB/darknet

----

* Paper **Scaled-YOLOv4 (CVPR 2021)**: https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Scaled-YOLOv4_Scaling_Cross_Stage_Partial_Network_CVPR_2021_paper.html

* source code Scaled-YOLOv4 - Pytorch (use to reproduce results): https://github.com/WongKinYiu/ScaledYOLOv4

----

### YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors

* **Paper**: https://arxiv.org/abs/2207.02696

* **source code - Pytorch (use to reproduce results):** https://github.com/WongKinYiu/yolov7

#### Datasets
- TACO Dataset
- Google Image Dataset V5

#### How to use on the command line

If you use `build.ps1` script or the makefile (Linux only) you will find `darknet` in the root directory.

If you use the deprecated Visual Studio solutions, you will find `darknet` in the directory `\build\darknet\x64`.

If you customize build with CMake GUI, darknet executable will be installed in your preferred folder.
