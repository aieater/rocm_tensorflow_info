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
  + ROCm(AMDGPU)-TensorFlow 1.10.0-x Python2.7/Python3.5/Python3.6 + UbuntuOS
  + CPU-TensorFlow 1.10.1 Python3.7 + MacOSX
- Lightweight ROCm-TensorFlow docker
  + ROCm-TensorFlow on GPUEater
  + ROCm-TensorFlow1.8 docker
<br>
<br>
<br>
<br>

### AMD Radeon GPU Driver + Computing Engine(ROCm 1.9.x) Installation for Python3
Python version 3.6 is the default python interpreter on Ubuntu 18.04. But as for Ubunt16.04, most of developers use Python version 3.5.
```
curl -sL http://install.aieater.com/setup_rocm | bash -
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



echo 'export HIP_VISIBLE_DEVICES=0' >> ~/.profile
echo 'export HCC_HOME=/opt/rocm/hcc' >> ~/.profile
echo 'export ROCM_HOME=/opt/rocm/bin' >> ~/.profile
echo 'export HIP_PATH=/opt/rocm/hip' >> ~/.profile
echo 'export PATH=/usr/local/bin:$HCC_HOME/bin:$HIP_PATH/bin:$ROCM_HOME:$PATH:/opt/rocm/opencl/bin/x86_64' >> ~/.profile
echo 'export LD_LIBRARY=$LD_LIBRARY:/opt/rocm/opencl/lib/x86_64' >> ~/.profile
echo 'export LC_ALL="en_US.UTF-8"' >> ~/.profile
echo 'export LC_CTYPE="en_US.UTF-8"' >> ~/.profile
echo 'export HSA_ENABLE_SDMA=0' >> ~/.profile

source ~/.profile


export HIP_PLATFORM=hcc
export PLATFORM=hcc

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

## Latest wheel binary


|  -  |  TYPE  |  OS  |  Python  |  TensorFlow  |  Install  |
| ---- | ---- | ---- | ---- | ---- | ---- |
|  AMD Radeon  |  GPU  |  Ubuntu |  3.6  |  1.10-latest  | sudo pip3 install http://install.aieater.com/gpueater/rocm/tensorflow-1.10.0-cp36-cp36m-linux_x86_64.whl |
|  AMD Radeon  |  GPU  |  Ubuntu  |  3.5  |  1.10-latest  | sudo pip3 install http://install.aieater.com/gpueater/rocm/tensorflow-1.10.0-cp35-cp35m-linux_x86_64.whl |
|  AMD Radeon  |  GPU  |  Ubuntu  |  3.6  |  1.10-rc2  | curl -sO http://install.aieater.com/gpueater/rocm/tensorflow-1.10.0-rc2-cp36-cp36m-linux_x86_64.whl tensorflow-1.10.0-cp36-cp36m-linux_x86_64.whl; sudo pip3 install tensorflow-1.10.0-cp36-cp36m-linux_x86_64.whl |
|  AMD Radeon  |  GPU  |  Ubuntu  |  3.5  |  1.10-rc2  | curl -sO http://install.aieater.com/gpueater/rocm/tensorflow-1.10.0-rc2-cp35-cp35m-linux_x86_64.whl tensorflow-1.10.0-cp35-cp35m-linux_x86_64.whl; sudo pip3 install tensorflow-1.10.0-cp35-cp35m-linux_x86_64.whl |
|  AMD Radeon  |  GPU  |  Ubuntu  |  3.6  |  1.10-rc0  | curl -sO http://install.aieater.com/gpueater/rocm/tensorflow-1.10.0-rc0-cp36-cp36m-linux_x86_64.whl tensorflow-1.10.0-cp36-cp36m-linux_x86_64.whl; sudo pip3 install tensorflow-1.10.0-cp36-cp36m-linux_x86_64.whl |
|  AMD Radeon  |  GPU  |  Ubuntu  |  3.6  |  1.8.0 | sudo pip3 install http://install.aieater.com/gpueater/rocm/tensorflow-1.8.0-cp36-cp36m-linux_x86_64.whl |
|  AMD Radeon  |  GPU  |  Ubuntu  |  2.7  |  1.8.0 | sudo pip install http://repo.radeon.com/rocm/misc/tensorflow/tensorflow-1.8.0-cp27-cp27mu-manylinux1_x86_64.whl |
|  AMD Radeon  |  GPU  |  Ubuntu  |  3.5  |  1.8.0 | sudo pip3 install http://repo.radeon.com/rocm/misc/tensorflow/tensorflow-1.8.0-cp35-cp35m-manylinux1_x86_64.whl |
|  -  |  CPU  |  MacOSX  |  3.7  |  1.10.1 | pip3 install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.10.1-py3-none-any.whl |
|  -  |  CPU  |  MacOSX  |  2.7  | latest | pip install tensorflow |
|  -  |  CPU  |  MacOSX  |  ~3.5  |  latest | pip3 install tensorflow |
|  -  |  CPU  |  Linux  |  2.7  | latest | pip install tensorflow |
|  -  |  CPU  |  Linux  |  ~3.5  | latest | pip3 install tensorflow |
|  NVIDIA  |  GPU  |  Linux  |  ~3.5  | latest | pip3 install tensorflow-gpu |
|  NVIDIA  |  GPU  |  Linux  |  2.7  | latest | pip install tensorflow-gpu |
|  ANY |  GPU  |  Linux  |  3.x  | unstable | pip3 install tf-nightly-gpu |
|  ANY |  GPU  |  Linux  |  2.x  | unstable | pip install tf-nightly-gpu |





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

## Docker

### TensorFlow 1.8 docker image on AMD Radeon GPU

<br>
<br>
<br>
<br>
###  # Recommended configurations
 OS: Ubuntu16.04.05+
 Kernel: 4.15+
 ROCm: 1.8.192+

### # AMD Radeon driver installation

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
sudo apt install -y libnuma-dev
sudo apt install -y rocm-dkms rocm-opencl-dev
sudo usermod -a -G video $LOGNAME
```

#### - Make sure to see AMD Radeon GPUs
```/opt/rocm/opencl/bin/x86_64/clinfo

ls -la /dev/kfd # AMD Kernel Fusion Driver
ls -la /dev/dri/ # Display and OpenCL file descriptors
```


###  # Docker-CE on Host

####  - Install docker-ce
 https://docs.docker.com/install/linux/docker-ce/ubuntu/

####  - Run a container with GPU driver's file descriptor
```docker run -it --device=/dev/kfd --device=/dev/dri --group-add video gpueater/rocm-tensorflow-1.8```



####  - Confirm GPUs on launched container

```sh
/opt/rocm/opencl/bin/x86_64/clinfo
```


Also see https://www.gpueater.com/help
