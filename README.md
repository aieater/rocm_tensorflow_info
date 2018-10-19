# AMD RadeonGPU ROCm-TensorFlow information
<br>
<br>
<br>
This README is intended to provide helpful information for Deep Learning developers with AMD ROCm.<br>
<br>
Unfortunately, AMD's official repository for ROCm sometimes includes old or missing information. Therefore, on this readme, we will endeavor to describe accurate information based on the knowledge gained by GPUEater infrastructure development and operation.<br>
<br>
<br>
<br>
<br>

- How to setup Radeon GPU Driver (ROCm) on Ubuntu16.04/18.04
- How to setup ROCm-Tensorflow on Ubuntu16.04/18.04
  + ROCm(AMDGPU)-TensorFlow 1.8 Python2.7/Python3.5 + UbuntuOS
  + ROCm(AMDGPU)-TensorFlow 1.1x.x Python2.7/Python3.5/Python3.6 + UbuntuOS
  + CPU-TensorFlow 1.1x.x Python3.7 + MacOSX
- Lightweight ROCm-TensorFlow docker
  + ROCm-TensorFlow on GPUEater
  + ROCm-TensorFlow1.11 docker
<br>
<br>
<br>
<br>

### Python3.5 + ROCm Driver + ROCm-TensorFlow1.11+ easy installer (Recommend)
```
curl -sL http://install.aieater.com/setup_rocm_tensorflow_p35 | bash -
```

<br>
<br>

### Make sure OpenCL AMD GPU devices

```
/opt/rocm/opencl/bin/x86_64/clinfo
```

```
johndoe@local:~$ /opt/rocm/opencl/bin/x86_64/clinfo

Number of platforms:				 1
  Platform Profile:				 FULL_PROFILE
  Platform Version:				 OpenCL 2.1 AMD-APP (2679.0)
  Platform Name:				 AMD Accelerated Parallel Processing
  Platform Vendor:				 Advanced Micro Devices, Inc.
  Platform Extensions:				 cl_khr_icd cl_amd_event_callback 


  Platform Name:				 AMD Accelerated Parallel Processing
Number of devices:				 1
  Device Type:					 CL_DEVICE_TYPE_GPU
  Vendor ID:					 1002h
  Board name:					 Device 687f
  Device Topology:				 PCI[ B#4, D#0, F#0 ]
  Max compute units:				 64
  Max work items dimensions:			 3
    Max work items[0]:				 1024
    Max work items[1]:				 1024
    Max work items[2]:				 1024
  Max work group size:				 256
  Preferred vector width char:			 4
  Preferred vector width short:			 2
  Preferred vector width int:			 1
  Preferred vector width long:			 1
  Preferred vector width float:			 1
  Preferred vector width double:		 1
  Native vector width char:			 4
  Native vector width short:			 2
  Native vector width int:			 1
  Native vector width long:			 1
  Native vector width float:			 1
  Native vector width double:			 1
  Max clock frequency:				 1630Mhz
  Address bits:					 64
  Max memory allocation:			 7287183769
  Image support:				 Yes
  Max number of images read arguments:		 128
  Max number of images write arguments:		 8
  Max image 2D width:				 16384
  Max image 2D height:				 16384
  Max image 3D width:				 2048
  Max image 3D height:				 2048
  Max image 3D depth:				 2048
  Max samplers within kernel:			 26751
  Max size of kernel argument:			 1024
  Alignment (bits) of base address:		 1024
  Minimum alignment (bytes) for any datatype:	 128
  Single precision floating point capability
    Denorms:					 Yes
    Quiet NaNs:					 Yes
    Round to nearest even:			 Yes
    Round to zero:				 Yes
    Round to +ve and infinity:			 Yes
    IEEE754-2008 fused multiply-add:		 Yes
  Cache type:					 Read/Write
  Cache line size:				 64
  Cache size:					 16384
  Global memory size:				 8573157376
  Constant buffer size:				 7287183769
  Max number of constant args:			 8
  Local memory type:				 Scratchpad
  Local memory size:				 65536
  Max pipe arguments:				 16
  Max pipe active reservations:			 16
  Max pipe packet size:				 2992216473
  Max global variable size:			 7287183769
  Max global variable preferred total size:	 8573157376
  Max read/write image args:			 64
  Max on device events:				 1024
  Queue on device max size:			 8388608
  Max on device queues:				 1
  Queue on device preferred size:		 262144
  SVM capabilities:				 
    Coarse grain buffer:			 Yes
    Fine grain buffer:				 Yes
    Fine grain system:				 No
    Atomics:					 No
  Preferred platform atomic alignment:		 0
  Preferred global atomic alignment:		 0
  Preferred local atomic alignment:		 0
  Kernel Preferred work group size multiple:	 64
  Error correction support:			 0
  Unified memory for Host and Device:		 0
  Profiling timer resolution:			 1
  Device endianess:				 Little
  Available:					 Yes
  Compiler available:				 Yes
  Execution capabilities:				 
    Execute OpenCL kernels:			 Yes
    Execute native function:			 No
  Queue on Host properties:				 
    Out-of-Order:				 No
    Profiling :					 Yes
  Queue on Device properties:				 
    Out-of-Order:				 Yes
    Profiling :					 Yes
  Platform ID:					 0x7efc84a47df0
  Name:						 gfx900
  Vendor:					 Advanced Micro Devices, Inc.
  Device OpenCL C version:			 OpenCL C 2.0 
  Driver version:				 2679.0 (HSA1.1,LC)
  Profile:					 FULL_PROFILE
  Version:					 OpenCL 1.2 
  Extensions:					 cl_khr_fp64 cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_int64_base_atomics cl_khr_int64_extended_atomics cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_fp16 cl_khr_gl_sharing cl_amd_device_attribute_query cl_amd_media_ops cl_amd_media_ops2 cl_khr_subgroups cl_khr_depth_images cl_amd_copy_buffer_p2p cl_amd_assembly_program 
  
```

<br>
<br>


#### Make sure TensorFlow AMD GPU devices

```
python3 -c "from tensorflow.python.client import device_lib;print(device_lib.list_local_devices())"
```

```
johndoe@local:~$ python3 -c "from tensorflow.python.client import device_lib;print(device_lib.list_local_devices())"

2018-10-11 21:17:31.776018: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2018-10-11 21:17:31.778774: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1524] Found device 0 with properties: 
name: Ellesmere [Radeon RX 470/480]
AMDGPU ISA: gfx803
memoryClockRate (GHz) 1.38
pciBusID 0000:01:00.0
Total memory: 8.00GiB
Free memory: 7.75GiB
2018-10-11 21:17:31.778788: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Adding visible gpu devices: 0
2018-10-11 21:17:31.778802: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1044] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-10-11 21:17:31.778806: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1050]      0 
2018-10-11 21:17:31.778810: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1063] 0:   N 
2018-10-11 21:17:31.778830: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1183] Created TensorFlow device (/device:GPU:0 with 7539 MB memory) -> physical GPU (device: 0, name: Ellesmere [Radeon RX 470/480], pci bus id: 0000:01:00.0)
[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 15907518430835446805
, name: "/device:GPU:0"
device_type: "GPU"
memory_limit: 7905424180
locality {
  bus_id: 2
  numa_node: 1
  links {
  }
}
incarnation: 9147052532367269657
physical_device_desc: "device: 0, name: Ellesmere [Radeon RX 470/480], pci bus id: 0000:01:00.0"
```

<br>
<br>
<br>
<br>
<br>
<br>

---------------------------------------------------------------------------------

<br>
<br>
<br>
<br>
<br>
<br>

## Optionals

<br>
<br>

### AMD Radeon GPU Driver + Computing Engine(ROCm 1.9.224+) Installation for Python3.5
```
curl -sL http://install.aieater.com/setup_rocm | bash -
```
or 
```
# Common
sudo apt update
sudo apt -y install software-properties-common curl wget # for add-apt-repository

# Python3.5
PYTHON35=false
if [[ `python3 --version` == *"3.5"* ]] ; then
  echo 'python3.5 -- yes'
  PYTHON35=true
else
  echo 'python3.5 -- no'
  PYTHON35=false
fi

if [ $PYTHON35 == 'true' ] ; then
  sudo apt install -y python3.5 python3.5-dev python3-pip
else
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update
  sudo apt install -y python3.5 python3.5-dev python3-pip
  sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 1
  sudo update-alternatives --set python3 /usr/bin/python3.5
  python3 --version
  curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
  sudo -H python3 /tmp/get-pip.py --force-reinstall
  sudo apt-get remove -y --purge python3-apt
fi




wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
sudo sh -c 'echo deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main > /etc/apt/sources.list.d/rocm.list'
sudo apt update
sudo apt install -y rocm-dkms rocm-libs miopen-hip cxlactivitylogger libnuma-dev
sudo usermod -a -G video $LOGNAME
/opt/rocm/opencl/bin/x86_64/clinfo

echo 'export ROCM_HOME=/opt/rocm' >> ~/.profile
echo 'export HCC_HOME=$ROCM_HOME/hcc' >> ~/.profile
echo 'export HIP_PATH=$ROCM_HOME/hip' >> ~/.profile
echo 'export PATH=/usr/local/bin:$HCC_HOME/bin:$HIP_PATH/bin:$ROCM_HOME/bin:$PATH:/opt/rocm/opencl/bin/x86_64' >> ~/.profile
echo 'export LD_LIBRARY=$LD_LIBRARY:/opt/rocm/opencl/lib/x86_64' >> ~/.profile
echo 'export LC_ALL="en_US.UTF-8"' >> ~/.profile
echo 'export LC_CTYPE="en_US.UTF-8"' >> ~/.profile


```

<br>
<br>

### ROCm-TensorFlow for Python3.5 installation via PyPI (You need to install ROCm-driver before TensorFlow.)
```
sudo pip3 uninstall -y tensorflow
sudo pip3 install --user tensorflow-rocm
```

<br>
<br>
<br>
<br>
<br>
<br>
<br>


### ~~AMD Radeon GPU Driver + Computing Engine(ROCm 1.9.x) Installation for Python3~~
(Deprecated)
~~Python version 3.6 is the default python interpreter on Ubuntu 18.04. But as for Ubunt16.04, most of developers use Python version 3.5.~~
```
curl -sL http://install.aieater.com/setup_rocm_old | bash -
```
or 
```
export PIP=pip3
export PYTHON=python3

sudo apt update
sudo apt upgrade -y
sudo apt install -y wget g++ cmake

mkdir -p ~/src


wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
sudo sh -c 'echo deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main > /etc/apt/sources.list.d/rocm.list'

sudo apt update
sudo apt install -y libnuma-dev
sudo apt install -y rocm-dkms rocm-opencl-dev
sudo usermod -a -G video $LOGNAME


/opt/rocm/opencl/bin/x86_64/clinfo


echo 'export ROCM_HOME=/opt/rocm' >> ~/.profile
echo 'export HCC_HOME=$ROCM_HOME/hcc' >> ~/.profile
echo 'export HIP_PATH=$ROCM_HOME/hip' >> ~/.profile
echo 'export PATH=/usr/local/bin:$HCC_HOME/bin:$HIP_PATH/bin:$ROCM_HOME/bin:$PATH:/opt/rocm/opencl/bin/x86_64' >> ~/.profile
echo 'export LD_LIBRARY=$LD_LIBRARY:/opt/rocm/opencl/lib/x86_64' >> ~/.profile
echo 'export LC_ALL="en_US.UTF-8"' >> ~/.profile
echo 'export LC_CTYPE="en_US.UTF-8"' >> ~/.profile


source ~/.profile


# Python3
sudo apt-get update && sudo apt-get install -y \
    $PYTHON-numpy \
    $PYTHON-dev \
    $PYTHON-wheel \
    $PYTHON-mock \
    $PYTHON-future \
    $PYTHON-pip \
    $PYTHON-yaml \
    $PYTHON-h5py \
    $PYTHON-setuptools && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*


# MIOpen
sudo apt-get update && \
    sudo apt-get install -y --allow-unauthenticated \
    rocm-dkms rocm-dev rocm-libs \
    rocm-device-libs \
    hsa-ext-rocr-dev hsakmt-roct-dev hsa-rocr-dev \
    rocm-opencl rocm-opencl-dev \
    rocm-utils \
    rocm-profiler cxlactivitylogger \
    miopen-hip miopengemm \

sudo $PIP install six numpy wheel cython pillow

```
<br>
<br>
<br>
<br>
<br>
<br>
<br>

## Latest wheel binaries

Finally, official ROCm-TensorFlow has registered to PyPI.
(But Python3.6+ version still only from source build.)
```
pip3 install --user tensorflow-rocm
```

|  -  |  TYPE  |  OS  |  Python  |  TensorFlow  | Vega | RX5xx |  Install  |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|  AMD Radeon  |  GPU  |  Ubuntu |  3.5  |  latest  |  | | pip3 install tensorflow-rocm |
|  AMD Radeon  |  GPU  |  Ubuntu |  3.6  |  1.11-rc1  |  | | pip3 install http://install.aieater.com/gpueater/rocm/tensorflow-1.11.0rc1-cp36-cp36m-linux_x86_64.whl |
|  AMD Radeon  |  GPU  |  Ubuntu |  3.6  |  1.10-latest  | | NG | pip3 install http://install.aieater.com/gpueater/rocm/tensorflow-1.10.0latest-cp36-cp36m-linux_x86_64.whl |
|  AMD Radeon  |  GPU  |  Ubuntu  |  3.6  |  1.10-rc2  | | NG | pip3 install http://install.aieater.com/gpueater/rocm/tensorflow-1.10.0rc2-cp36-cp36m-linux_x86_64.whl  |
|  AMD Radeon  |  GPU  |  Ubuntu  |  3.5  |  1.10-rc2  | | NG | pip3 install http://install.aieater.com/gpueater/rocm/tensorflow-1.10.0rc2-cp35-cp35m-linux_x86_64.whl  |
|  AMD Radeon  |  GPU  |  Ubuntu  |  3.6  |  1.10-rc0  | | NG | pip3 install http://install.aieater.com/gpueater/rocm/tensorflow-1.10.0rc0-cp36-cp36m-linux_x86_64.whl  |
|  AMD Radeon  |  GPU  |  Ubuntu  |  3.6  |  1.8.0 | | NG | pip3 install http://install.aieater.com/gpueater/rocm/tensorflow-1.8.0-cp36-cp36m-linux_x86_64.whl |
|  AMD Radeon  |  GPU  |  Ubuntu  |  2.7  |  1.8.0 | | | pip install http://repo.radeon.com/rocm/misc/tensorflow/tensorflow-1.8.0-cp27-cp27mu-manylinux1_x86_64.whl |
|  AMD Radeon  |  GPU  |  Ubuntu  |  3.5  |  1.8.0 | | | pip3 install http://repo.radeon.com/rocm/misc/tensorflow/tensorflow-1.8.0-cp35-cp35m-manylinux1_x86_64.whl |
|  -  |  CPU  |  MacOSX  |  3.7  |  1.10.1 | | | pip3 install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.10.1-py3-none-any.whl |
|  -  |  CPU  |  MacOSX  |  2.7  | latest | | | pip install tensorflow |
|  -  |  CPU  |  MacOSX  |  ~3.5  |  latest | | | pip3 install tensorflow |
|  -  |  CPU  |  Linux  |  2.7  | latest | | | pip install tensorflow |
|  -  |  CPU  |  Linux  |  ~3.5  | latest | | | pip3 install tensorflow |
|  NVIDIA  |  GPU  |  Linux  |  ~3.5  | latest | | | pip3 install tensorflow-gpu |
|  NVIDIA  |  GPU  |  Linux  |  2.7  | latest | | | pip install tensorflow-gpu |
|  ANY |  GPU  |  Linux  |  3.x  | unstable | | | pip3 install tf-nightly-gpu |
|  ANY |  GPU  |  Linux  |  2.x  | unstable | | | pip install tf-nightly-gpu |




### RX580 issue
RX580 has something problem and unstable. (2018/10/2)
Getting GPU name was mistaken as Ellesmere "[Radeon RX 470/480]".
Vega64, 56, FE edition is stable on ROCm-1.9.211 + ROCm-TensorFlow1.8+
```
johndoe@gpueater.local:~/projects/models/tutorials/image/cifar10$ python3 cifar10_multi_gpu_train.py 
Filling queue with 20000 CIFAR images before starting to train. This will take a few minutes.
2018-10-02 14:48:22.955238: W tensorflow/stream_executor/rocm/rocm_driver.cc:404] creating context when one is currently active; existing: 0x7fe4de1c9d30
2018-10-02 14:48:22.955859: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1451] Found device 0 with properties: 
name: Ellesmere [Radeon RX 470/480]
AMDGPU ISA: gfx803
memoryClockRate (GHz) 1.38
pciBusID 0000:01:00.0
Total memory: 8.00GiB
Free memory: 7.75GiB
2018-10-02 14:48:22.955874: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1562] Adding visible gpu devices: 0
2018-10-02 14:48:22.955888: I tensorflow/core/common_runtime/gpu/gpu_device.cc:989] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-10-02 14:48:22.955892: I tensorflow/core/common_runtime/gpu/gpu_device.cc:995]      0 
2018-10-02 14:48:22.955895: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1008] 0:   N 
2018-10-02 14:48:22.955921: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1124] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 7539 MB memory) -> physical GPU (device: 0, name: Ellesmere [Radeon RX 470/480], pci bus id: 0000:01:00.0)
terminate called after throwing an instance of 'std::runtime_error'
  what():  No device code available for function: _ZN7rocprim6detail19block_reduce_kernelILj256ELj4ELb0EfNS_18transform_iteratorIPfN10tensorflow10squareHalfIfEEfEES3_fN6hipcub3SumEEEvT3_mT4_T5_T6_

```




### A installation memo of the latest version of TensorFlow.
ROCm tensorflow-upstream  (https://github.com/ROCmSoftwarePlatform/tensorflow-upstream)

```
mkdir -p ~/src
cd ~/src
BAZEL=0.15.0
TENSORFLOW_BRANCH=v1.10.0-rocm-rc2
rm -rf ~/.bazel ~/.cache/bazel
if test -e "bazel-$BAZEL-installer-linux-x86_64.sh"; then
  echo "bazel-$BAZEL-installer-linux-x86_64.sh found."
else
  echo "bazel-$BAZEL-installer-linux-x86_64.sh NOT found."
  wget https://github.com/bazelbuild/bazel/releases/download/$BAZEL/bazel-$BAZEL-installer-linux-x86_64.sh
fi
chmod +x bazel-$BAZEL-installer-linux-x86_64.sh
./bazel-$BAZEL-installer-linux-x86_64.sh --user
source ~/.bazel/bin/bazel-complete.bash
export PATH=~/.bazel/bin:$PATH
sudo apt-get install -y openjdk-8-jdk
git clone https://github.com/ROCmSoftwarePlatform/tensorflow-upstream.git
cd tensorflow-upstream
git pull origin $TENSORFLOW_BRANCH
# ./build_rocm_python # 2.7
sudo pip3 uninstall -y tensorflow
 ./build_rocm_python3 & # 3.x
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
cp -f /tmp/tensorflow_pkg/* ~/src/
pip3 install ~/src/tensorflow*.whl


python3 -c "from tensorflow.python.client import device_lib;device_lib.list_local_devices()"
```
<br>
<br>
<br>

#### Bazel

```
tensorflow-1.10.1	CPU	2.7, 3.3-3.6	GCC 4.8	Bazel 0.15.0	N/A	N/A
tensorflow_gpu-1.10.1	GPU	2.7, 3.3-3.6	GCC 4.8	Bazel 0.15.0	7	9
tensorflow-1.9.0	CPU	2.7, 3.3-3.6	GCC 4.8	Bazel 0.11.0	N/A	N/A
tensorflow_gpu-1.9.0	GPU	2.7, 3.3-3.6	GCC 4.8	Bazel 0.11.0	7	9
tensorflow-1.8.0	CPU	2.7, 3.3-3.6	GCC 4.8	Bazel 0.10.0	N/A	N/A
tensorflow_gpu-1.8.0	GPU	2.7, 3.3-3.6	GCC 4.8	Bazel 0.9.0	7	9
```


<br>
<br>
<br>
<br>

### Show devices
```
python3 -c "from tensorflow.python.client import device_lib;device_lib.list_local_devices()"
```
```
2018-09-05 13:21:43.760601: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1520] Found device 0 with properties:
name: Vega [Radeon RX Vega]
AMDGPU ISA: gfx900
memoryClockRate (GHz) 1.63
pciBusID 0000:04:00.0
Total memory: 7.98GiB
Free memory: 7.73GiB
2018-09-05 13:21:43.760632: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1631] Adding visible gpu devices: 0
2018-09-05 13:21:43.760644: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1040] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-09-05 13:21:43.760649: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1046]      0
2018-09-05 13:21:43.760653: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1059] 0:   N
2018-09-05 13:21:43.760697: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1179] Created TensorFlow device (/device:GPU:0 with 7524 MB memory) -> physical GPU (device: 0, name: Vega [Radeon RX Vega], pci bus id: 0000:04:00.0)
```

<br>
<br>


## How to confirm Radeon GPU's memory usage on GPUEater instance.

```
johndoe@gpueater.local:~$ curl -O http://install.aieater.com/gpueater/rocm/gpueater-smi
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 45447  100 45447    0     0  1643k      0 --:--:-- --:--:-- --:--:-- 1643k

johndoe@gpueater.local:~$ chmod +x ./gpueater-smi
johndoe@gpueater.local:~$ ./gpueater-smi


====================    ROCm System Management Interface    ====================
================================================================================
 GPU  Temp    AvgPwr   SCLK     MCLK     Fan      Perf    SCLK OD    MCLK OD  USED MEM
  0   48c     4.0W     852Mhz   167Mhz   35.69%   auto      0%         0%       7619MB
================================================================================
====================           End of ROCm SMI Log          ====================

johndoe@gpueater.local:~$ mv gpueater-smi `which rocm-smi`
```

<br>
<br>
<br>
<br>
<br>




-----------------------------------------------------
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

------------------------------------------------------

<br>
<br>
<br>

## Recommend pip3 packages installation

- tensorflow-rocm
- keras
- cython
- numpy
- moviepy
- requests
- sklearn
- cairocffi
- matplotlib
- editdistance
- pandas
- portpicker
- h5py
- PIL
- darkflow
- cv2
- jupyter



```
curl -sL http://install.aieater.com/setup_ml_submod | bash -
```
or
```
curl -O http://install.aieater.com/check_mod.py
python3 check_mod.py
chmod +x install.sh
./install.sh
```

<br>
<br>
<br>
<br>

---------------------------------------------------------------

## Examples

### YoloV2

https://s3-ap-northeast-1.amazonaws.com/gpueater/examples/deep_learning_yolo_v2.tar.gz
 
```
wget https://s3-ap-northeast-1.amazonaws.com/gpueater/examples/deep_learning_yolo_v2.tar.gz
tar xf deep_learning_yolo_v2.tar.gz
cd deep_learning_yolo_v2
python3 main.py image.jpg
```

 References
  - https://github.com/thtrieu/darkflow
  - https://pjreddie.com/darknet/yolo/

 
### YoloV3

https://s3-ap-northeast-1.amazonaws.com/gpueater/examples/deep_learning_yolo_v3.tar.gz

```
wget https://s3-ap-northeast-1.amazonaws.com/gpueater/examples/deep_learning_yolo_v3.tar.gz
tar xf deep_learning_yolo_v3.tar.gz
cd deep_learning_yolo_v3
python3 yolo.py image.jpg
```
 References
  - https://github.com/qqwweee/keras-yolo3
  - https://pjreddie.com/darknet/yolo/



---------------------------------------------------------------


<br>
<br>
<br>
<br>


## Docker

#### Available images

https://hub.docker.com/r/gpueater/ubuntu16-rocm-1.9.211-tensorflow-1.11.0/

https://hub.docker.com/r/gpueater/rocm-tensorflow-1.8/

### Latest

```
docker run -it --device=/dev/kfd --device=/dev/dri gpueater/ubuntu16-rocm-1.9.211-tensorflow-1.11.0
```

### Old images

```
docker run -it --device=/dev/kfd --device=/dev/dri gpueater/rocm-tensorflow-1.8
```

<br>
<br>
<br>

## ROCm1.9.211+TensorFlow 1.11.0 image for AMD Radeon GPU




###  # Recommended environment of host

 OS: Ubuntu16.04.05+
 Kernel: 4.15+
 ROCm: 1.9.211+

### # AMD Radeon driver installation on Host

#### - Update linux kernel

 \* If you already used the GPUEater AMD GPU instance, the following command is not required.

```sh
sudo apt update
sudo apt upgrade -y
sudo apt install -y linux-generic-hwe-16.04
sudo reboot
```

#### - Install AMD GPU Driver (ROCm)

 \* If you already used the GPUEater AMD GPU instance, the following command is not required.

```sh
sudo apt install -y wget
wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
sudo sh -c 'echo deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main > /etc/apt/sources.list.d/rocm.list'
sudo apt install -y libnuma-dev rocm-dkms
sudo usermod -a -G video $LOGNAME
```

#### - Make sure to see AMD Radeon GPUs.

```/opt/rocm/opencl/bin/x86_64/clinfo

ls -la /dev/kfd # AMD Kernel Fusion Driver
ls -la /dev/dri/ # Display and OpenCL file descriptors
```


###  # Docker-CE on Host

####  - Install docker-ce

https://docs.docker.com/install/linux/docker-ce/ubuntu/

####  - Run a container with GPU driver file descriptor.

```docker run -it --device=/dev/kfd --device=/dev/dri gpueater/ubuntu16-rocm-1.9.211-tensorflow-1.11.0```



####  - Make sure GPUs in launched container

```sh
/opt/rocm/opencl/bin/x86_64/clinfo

Number of platforms:				 1
  Platform Profile:				 FULL_PROFILE
  Platform Version:				 OpenCL 2.1 AMD-APP.internal (2574.0)
  Platform Name:				 AMD Accelerated Parallel Processing
  Platform Vendor:				 Advanced Micro Devices, Inc.
  Platform Extensions:				 cl_khr_icd cl_amd_object_metadata cl_amd_event_callback 


  Platform Name:				 AMD Accelerated Parallel Processing
Number of devices:				 1
  Device Type:					 CL_DEVICE_TYPE_GPU
  Vendor ID:					 1002h
  Board name:					 Device 687f
  Device Topology:				 PCI[ B#3, D#0, F#0 ]
  Max compute units:				 56
  Max work items dimensions:			 3
    Max work items[0]:				 1024
    Max work items[1]:				 1024
    Max work items[2]:				 1024
  Max work group size:				 256
  Preferred vector width char:			 4
  Preferred vector width short:			 2
  Preferred vector width int:			 1
  Preferred vector width long:			 1
  Preferred vector width float:			 1
  Preferred vector width double:		 1
  Native vector width char:			 4
  Native vector width short:			 2
  Native vector width int:			 1
  Native vector width long:			 1
  Native vector width float:			 1
  Native vector width double:			 1
  Max clock frequency:				 1622Mhz
  Address bits:					 64
  Max memory allocation:			 7287183769
  Image support:				 Yes
  Max number of images read arguments:		 128
  Max number of images write arguments:		 8
  Max image 2D width:				 16384
  Max image 2D height:				 16384
  Max image 3D width:				 2048
  Max image 3D height:				 2048
  Max image 3D depth:				 2048
  Max samplers within kernel:			 26751
  Max size of kernel argument:			 1024
  Alignment (bits) of base address:		 1024
  Minimum alignment (bytes) for any datatype:	 128
  Single precision floating point capability
    Denorms:					 Yes
    Quiet NaNs:					 Yes
    Round to nearest even:			 Yes
    Round to zero:				 Yes
    Round to +ve and infinity:			 Yes
    IEEE754-2008 fused multiply-add:		 Yes
  Cache type:					 Read/Write
  Cache line size:				 64
  Cache size:					 16384
  Global memory size:				 8573157376
  Constant buffer size:				 7287183769
  Max number of constant args:			 8
  Local memory type:				 Scratchpad
  Local memory size:				 65536
  Max pipe arguments:				 16
  Max pipe active reservations:			 16
  Max pipe packet size:				 2992216473
  Max global variable size:			 7287183769
  Max global variable preferred total size:	 8573157376
  Max read/write image args:			 64
  Max on device events:				 0
  Queue on device max size:			 0
  Max on device queues:				 0
  Queue on device preferred size:		 0
  SVM capabilities:				 
    Coarse grain buffer:			 Yes
    Fine grain buffer:				 Yes
    Fine grain system:				 No
    Atomics:					 No
  Preferred platform atomic alignment:		 0
  Preferred global atomic alignment:		 0
  Preferred local atomic alignment:		 0
  Kernel Preferred work group size multiple:	 64
  Error correction support:			 0
  Unified memory for Host and Device:		 0
  Profiling timer resolution:			 1
  Device endianess:				 Little
  Available:					 Yes
  Compiler available:				 Yes
  Execution capabilities:				 
    Execute OpenCL kernels:			 Yes
    Execute native function:			 No
  Queue on Host properties:				 
    Out-of-Order:				 No
    Profiling :					 Yes
  Queue on Device properties:				 
    Out-of-Order:				 No
    Profiling :					 No
  Platform ID:					 0x7f248a35f270
  Name:						 gfx900
  Vendor:					 Advanced Micro Devices, Inc.
  Device OpenCL C version:			 OpenCL C 2.0 
  Driver version:				 2574.0 (HSA1.1,LC)
  Profile:					 FULL_PROFILE
  Version:					 OpenCL 1.2 
  Extensions:					 cl_khr_fp64 cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_int64_base_atomics cl_khr_int64_extended_atomics cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_fp16 cl_khr_gl_sharing cl_amd_device_attribute_query cl_amd_media_ops cl_amd_media_ops2 cl_khr_subgroups cl_khr_depth_images cl_amd_copy_buffer_p2p cl_amd_assembly_program 
```


Also see https://www.gpueater.com/help

Also see https://github.com/aieater/rocm_tensorflow_info



