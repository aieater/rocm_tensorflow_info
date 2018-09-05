# AMD RadeonGPU ROCm-TensorFlow information and memo


### AMD Radeon GPU computing driver ROCm 1.8.x installation for Python3
Ubuntu18.04 default is Python3.6, but Ubunt16.04 is still Python3.5. 
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


cd ~/src
git clone https://github.com/ROCmSoftwarePlatform/hcRNG.git && cd hcRNG
./install.sh

cd ~/src
git clone https://github.com/ROCmSoftwarePlatform/hcFFT.git && cd hcFFT
./install.sh

sudo $PIP install six numpy wheel cython pillow
```


### ROCm-TensorFlow1.10.0-rc0 binary for Python3.6/Ubuntu18.04
```
sudo pip3 install http://install.aieater.com/gpueater/rocm/tensorflow-1.10.0-rc0-cp36-cp36m-linux_x86_64.whl
```

### ROCm-TensorFlow1.8.0 binary for Python2.7
```
sudo pip install http://repo.radeon.com/rocm/misc/tensorflow/tensorflow-1.8.0-cp27-cp27mu-manylinux1_x86_64.whl
```
### ROCm-TensorFlow1.8.0 binary for Python3.5 
```
sudo pip3 install http://repo.radeon.com/rocm/misc/tensorflow/tensorflow-1.8.0-cp35-cp35m-manylinux1_x86_64.whl
```

### Latest version building and installation memo.
ROCm tensorflow-upstream  (https://github.com/ROCmSoftwarePlatform/tensorflow-upstream)

```
BAZEL=0.15.0
TENSORFLOW_BRANCH=r1.10-rocm
rm -rf ~/.bazel ~/.cache/bazel
wget https://github.com/bazelbuild/bazel/releases/download/$BAZEL/bazel-$BAZEL-installer-linux-x86_64.sh
chmod +x bazel-$BAZEL-installer-linux-x86_64.sh
./bazel-$BAZEL-installer-linux-x86_64.sh --user
sudo apt-get install openjdk-8-jdk
	
git clone https://github.com/ROCmSoftwarePlatform/tensorflow-upstream.git
cd tensorflow-upstream
git pull origin $TENSORFLOW_BRANCH
	
# ./build_rocm_python # 2.7
./build_rocm_python3 # 3.x
	
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
cd /tmp/tensorflow_pkg
sudo pip3 install tensorflow-1.10.0-cp36-cp36m-linux_x86_64.whl
```


### Show devices
```
python3 -c "from tensorflow.python.client import device_lib;device_lib.list_local_devices()"
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


-----------------------------------------------------


## TensorFlow 1.8 docker image on AMD Radeon GPU




###  # Recommended environment of host
 OS: Ubuntu16.04.05+
 Kernel: 4.15+
 ROCm: 1.8.192+

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
sudo apt install -y libnuma-dev
sudo apt install -y rocm-dkms rocm-opencl-dev
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
```docker run -it --device=/dev/kfd --device=/dev/dri --group-add video gpueater/rocm-tensorflow-1.8```



####  - Make sure GPUs in launched container

```sh
/opt/rocm/opencl/bin/x86_64/clinfo
```





