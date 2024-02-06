# Trash Detection Using Yolov: ML and CV

## (Neural Networks for Object Detection)

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

### Requirements for Windows, Linux and macOS

- **CMake >= 3.18**: https://cmake.org/download/
- **Powershell** (already installed on windows): https://docs.microsoft.com/en-us/powershell/scripting/install/installing-powershell
- **CUDA >= 10.2**: https://developer.nvidia.com/cuda-toolkit-archive (on Linux do [Post-installation Actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions))
- **OpenCV >= 2.4**: use your preferred package manager (brew, apt), build from source using [vcpkg](https://github.com/Microsoft/vcpkg) or download from [OpenCV official site](https://opencv.org/releases.html) (on Windows set system variable `OpenCV_DIR` = `C:\opencv\build` - where are the `include` and `x64` folders [image](https://user-images.githubusercontent.com/4096485/53249516-5130f480-36c9-11e9-8238-a6e82e48c6f2.png))
- **cuDNN >= 8.0.2** https://developer.nvidia.com/rdp/cudnn-archive (on **Linux** follow steps described here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installlinux-tar , on **Windows** follow steps described here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installwindows)
- **GPU with CC >= 3.0**: https://en.wikipedia.org/wiki/CUDA#GPUs_supported

#### Datasets

- MS COCO: use `./scripts/get_coco_dataset.sh` to get labeled MS COCO detection dataset
- OpenImages: use `python ./scripts/get_openimages_dataset.py` for labeling train detection dataset
- Pascal VOC: use `python ./scripts/voc_label.py` for labeling Train/Test/Val detection datasets
- ILSVRC2012 (ImageNet classification): use `./scripts/get_imagenet_train.sh` (also `imagenet_label.sh` for labeling valid set)
- German/Belgium/Russian/LISA/MASTIF Traffic Sign Datasets for Detection - use this parser: https://github.com/angeligareta/Datasets2Darknet#detection-task
- List of other datasets: https://github.com/AlexeyAB/darknet/tree/master/scripts#datasets

#### How to use on the command line

If you use `build.ps1` script or the makefile (Linux only) you will find `darknet` in the root directory.

If you use the deprecated Visual Studio solutions, you will find `darknet` in the directory `\build\darknet\x64`.

If you customize build with CMake GUI, darknet executable will be installed in your preferred folder.

- Yolo v4 COCO - **image**: `./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights -thresh 0.25`
- **Output coordinates** of objects: `./darknet detector test cfg/coco.data yolov4.cfg yolov4.weights -ext_output dog.jpg`
- Yolo v4 COCO - **video**: `./darknet detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights -ext_output test.mp4`
- Yolo v4 COCO - **WebCam 0**: `./darknet detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights -c 0`
- Yolo v4 COCO for **net-videocam** - Smart WebCam: `./darknet detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights http://192.168.0.80:8080/video?dummy=param.mjpg`
- Yolo v4 - **save result videofile res.avi**: `./darknet detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights test.mp4 -out_filename res.avi`
- Yolo v3 **Tiny** COCO - video: `./darknet detector demo cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights test.mp4`
- **JSON and MJPEG server** that allows multiple connections from your soft or Web-browser `ip-address:8070` and 8090: `./darknet detector demo ./cfg/coco.data ./cfg/yolov3.cfg ./yolov3.weights test50.mp4 -json_port 8070 -mjpeg_port 8090 -ext_output`
- Yolo v3 Tiny **on GPU #1**: `./darknet detector demo cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights -i 1 test.mp4`
- Alternative method Yolo v3 COCO - image: `./darknet detect cfg/yolov4.cfg yolov4.weights -i 0 -thresh 0.25`
- Train on **Amazon EC2**, to see mAP & Loss-chart using URL like: `http://ec2-35-160-228-91.us-west-2.compute.amazonaws.com:8090` in the Chrome/Firefox (**Darknet should be compiled with OpenCV**):
    `./darknet detector train cfg/coco.data yolov4.cfg yolov4.conv.137 -dont_show -mjpeg_port 8090 -map`
- 186 MB Yolo9000 - image: `./darknet detector test cfg/combine9k.data cfg/yolo9000.cfg yolo9000.weights`
- Remember to put data/9k.tree and data/coco9k.map under the same folder of your app if you use the cpp api to build an app
- To process a list of images `data/train.txt` and save results of detection to `result.json` file use:
    `./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights -ext_output -dont_show -out result.json < data/train.txt`
- To process a list of images `data/train.txt` and save results of detection to `result.txt` use:
    `./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights -dont_show -ext_output < data/train.txt > result.txt`
- To process a video and output results to a json file use: `darknet.exe detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights file.mp4 -dont_show -json_file_output results.json`
- Pseudo-labelling - to process a list of images `data/new_train.txt` and save results of detection in Yolo training format for each image as label `<image_name>.txt` (in this way you can increase the amount of training data) use:
    `./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights -thresh 0.25 -dont_show -save_labels < data/new_train.txt`
- To calculate anchors: `./darknet detector calc_anchors data/obj.data -num_of_clusters 9 -width 416 -height 416`
- To check accuracy mAP@IoU=50: `./darknet detector map data/obj.data yolo-obj.cfg backup\yolo-obj_7000.weights`
- To check accuracy mAP@IoU=75: `./darknet detector map data/obj.data yolo-obj.cfg backup\yolo-obj_7000.weights -iou_thresh 0.75`

##### For using network video-camera mjpeg-stream with any Android smartphone

1. Download for Android phone mjpeg-stream soft: IP Webcam / Smart WebCam

    - Smart WebCam - preferably: https://play.google.com/store/apps/details?id=com.acontech.android.SmartWebCam2
    - IP Webcam: https://play.google.com/store/apps/details?id=com.pas.webcam

2. Connect your Android phone to the computer by WiFi (through a WiFi-router) or USB
3. Start Smart WebCam on your phone
4. Replace the address below, shown in the phone application (Smart WebCam) and launch:

- Yolo v4 COCO-model: `./darknet detector demo data/coco.data yolov4.cfg yolov4.weights http://192.168.0.80:8080/video?dummy=param.mjpg -i 0`

### How to compile on Linux/macOS (using `CMake`)

The `CMakeLists.txt` will attempt to find installed optional dependencies like CUDA, cudnn, ZED and build against those. It will also create a shared object library file to use `darknet` for code development.

To update CMake on Ubuntu, it's better to follow guide here: https://apt.kitware.com/ or https://cmake.org/download/

```bash
git clone https://github.com/AlexeyAB/darknet
cd darknet
mkdir build_release
cd build_release
cmake ..
cmake --build . --target install --parallel 8
```

### Using also PowerShell

Install: `Cmake`, `CUDA`, `cuDNN` [How to install dependencies](#requirements)

Install powershell for your OS (Linux or MacOS) ([guide here](https://docs.microsoft.com/en-us/powershell/scripting/install/installing-powershell)).

Open PowerShell type these commands

```PowerShell
git clone https://github.com/AlexeyAB/darknet
cd darknet
./build.ps1 -UseVCPKG -EnableOPENCV -EnableCUDA -EnableCUDNN
```

- remove options like `-EnableCUDA` or `-EnableCUDNN` if you are not interested into
- remove option `-UseVCPKG` if you plan to manually provide OpenCV library to darknet or if you do not want to enable OpenCV integration
- add option `-EnableOPENCV_CUDA` if you want to build OpenCV with CUDA support - very slow to build! (requires `-UseVCPKG`)

If you open the `build.ps1` script at the beginning you will find all available switches.

### How to compile on Linux (using `make`)

Just do `make` in the darknet directory. (You can try to compile and run it on Google Colab in cloud [link](https://colab.research.google.com/drive/12QusaaRj_lUwCGDvQNfICpa7kA7_a2dE) (press «Open in Playground» button at the top-left corner) and watch the video [link](https://www.youtube.com/watch?v=mKAEGSxwOAY) )
Before make, you can set such options in the `Makefile`: [link](https://github.com/AlexeyAB/darknet/blob/9c1b9a2cf6363546c152251be578a21f3c3caec6/Makefile#L1)

- `GPU=1` to build with CUDA to accelerate by using GPU (CUDA should be in `/usr/local/cuda`)
- `CUDNN=1` to build with cuDNN v5-v7 to accelerate training by using GPU (cuDNN should be in `/usr/local/cudnn`)
- `CUDNN_HALF=1` to build for Tensor Cores (on Titan V / Tesla V100 / DGX-2 and later) speedup Detection 3x, Training 2x
- `OPENCV=1` to build with OpenCV 4.x/3.x/2.4.x - allows to detect on video files and video streams from network cameras or web-cams
- `DEBUG=1` to build debug version of Yolo
- `OPENMP=1` to build with OpenMP support to accelerate Yolo by using multi-core CPU
- `LIBSO=1` to build a library `darknet.so` and binary runnable file `uselib` that uses this library. Or you can try to run so `LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH ./uselib test.mp4` How to use this SO-library from your own code - you can look at C++ example: https://github.com/AlexeyAB/darknet/blob/master/src/yolo_console_dll.cpp
    or use in such a way: `LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH ./uselib data/coco.names cfg/yolov4.cfg yolov4.weights test.mp4`
- `ZED_CAMERA=1` to build a library with ZED-3D-camera support (should be ZED SDK installed), then run
    `LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH ./uselib data/coco.names cfg/yolov4.cfg yolov4.weights zed_camera`
- You also need to specify for which graphics card the code is generated. This is done by setting `ARCH=`. If you use a newer version than CUDA 11 you further need to edit line 20 from Makefile and remove `-gencode arch=compute_30,code=sm_30 \` as Kepler GPU support was dropped in CUDA 11. You can also drop the general `ARCH=` and just uncomment `ARCH=` for your graphics card.

### How to compile on Windows (using `CMake`)

Requires:

- MSVC: https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community
- CMake GUI: `Windows win64-x64 Installer`https://cmake.org/download/
- Download Darknet zip-archive with the latest commit and uncompress it: [master.zip](https://github.com/AlexeyAB/darknet/archive/master.zip)

In Windows:

- Start (button) -> All programs -> CMake -> CMake (gui) ->

- [look at image](https://habrastorage.org/webt/pz/s1/uu/pzs1uu4heb7vflfcjqn-lxy-aqu.jpeg) In CMake: Enter input path to the darknet Source, and output path to the Binaries -> Configure (button) -> Optional platform for generator: `x64`  -> Finish -> Generate -> Open Project ->

- in MS Visual Studio: Select: x64 and Release -> Build -> Build solution

- find the executable file `darknet.exe` in the output path to the binaries you specified

![x64 and Release](https://habrastorage.org/webt/ay/ty/f-/aytyf-8bufe7q-16yoecommlwys.jpeg)

