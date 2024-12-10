# A2C-LoVQI: A2C para implantar VANTs como gateways LoRaWAN com foco na Qualidade de Serviço IoT

### Soluções baseadas em aprendizado por reforço profundo para implantar VANTs como gateways LoRaWAN com foco na Qualidade de Serviço IoT 

This repository contains the code for the paper **"Soluções baseadas em aprendizado por reforço profundo para implantar VANTs como gateways LoRaWAN com foco na Qualidade de Serviço IoT"** submitted to the  **[SBrT 2024](https://sbrt2024.sbrt.org.br/)**. 

# How to use this repository

## 1. Prepare the environment

### 1.1 Get NS-3

NS-3 is a free, open-source project aiming to build a discrete-event
network simulator targeted for simulation research and education.
Get binaries from the [official website](https://www.nsnam.org/releases/ns-3-41/).

Download a source archive of |ns3| to a location on your file
system (usually somewhere under your home directory).

   ```bash
    wget -O <local-path> https://www.nsnam.org/releases/ns-allinone-3.41.tar.bz2 
   ```

Tar the file:

   ```bash
    tar -xjfv <local-path>/ns-allinone-3.41.tar.bz2
   ```

### 1.2 Build NS-3

#### Pre-requisites

Make sure that your system has these prerequisites. If not, install them using the following commands:

   ```bash
        # update the system
        sudo apt update && sudo apt upgrade -y
        # minimal requirements for release 3.37 and later
        sudo apt install g++ python3 cmake ninja-build git ccache
        # for the minimal requirements for Python visualizer and bindings
        python3 -m pip install --user cppyy
        sudo apt install gir1.2-goocanvas-2.0 python3-gi python3-gi-cairo python3-pygraphviz gir1.2-gtk-3.0 ipython3 
        # Additional minimal requirements for Python (development): 
        sudo apt install python3-setuptools 
        # Netanim animator:
        sudo apt install qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools
        # Netanim for Ubuntu 20.04 and later, the single 'qt5-default' package suffices
        sudo apt install qt5-default
        # Support for MPI-based distributed emulation
        sudo apt install openmpi-bin openmpi-common openmpi-doc libopenmpi-dev
        # Debugging:
        sudo apt install gdb valgrind 
        # To read pcap packet traces
        sudo apt install tcpdump
        # Database support for statistics framework
        sudo apt install sqlite sqlite3 libsqlite3-dev
        # To experiment with virtual machines and ns-3
        sudo apt install vtun lxc uml-utilities
   ```

#### Build NS-3

Go to the directory where you extracted the tarball and run the following commands:

   ```bash
       cd ns-allinone-3.41
       ./build.py --enable-examples --enable-tests
   ```

### 1.3 Install additional third-party modules

#### 1.3.1 Install the SIGNETLab/LoRaWAN module

Clone the LoRaWAN module on the ns-3.41/src directory:

   ```bash
       git clone https://github.com/signetlabdei/lorawan <local-path>/ns-3.41/src/lorawan
   ```

#### 1.3.2 Install the CTTC/5G-LENA module

##### 1.3.2.1 NR module pre-requisites

   ```bash
      # Install libc6-dev (it provides semaphore.h header file).  The Ubuntu package name is:
      sudo apt install libc6-dev
      # Install sqlite (enables optional examples lena-lte-comparison, cttc-nr-3gpp-calibration and cttc-realistic-beamforming):
      sudo apt install sqlite sqlite3 libsqlite3-dev
      # Install eigen3 (enables optional MIMO features):
      sudo apt-get install libeigen3-dev
   ```

##### 1.3.2.2 Clone the CTTC-nr module, on the ns-3.41/contrib directory:

```bash
    git clone https://gitlab.com/cttc-lena/nr.git <local-path>/ns-3.41/contrib/nr
    cd <local-path>/ns-3.41/contrib/nr
    git checkout 5g-lena-v3.0.y
   ```

#### 1.3.3 Install the TKN/ns3-gym module

##### 1.3.3.1 OpenGym module pre-requisites

   ```bash
      # minimal requirements for C++:
      sudo apt install gcc g++ python3 python3-pip cmake
      # Install ZMQ, Protocol Buffers and pkg-config libs:
      
      sudo apt install libzmq5 libzmq3-dev
      apt-get install libprotobuf-dev
      apt-get install protobuf-compiler
      apt-get install pkg-config
   ```

##### 1.3.3.2 Clone the ns3-gym module, on the ns-3.41/contrib directory:

OpenAI Gym is a toolkit for reinforcement learning (RL) and 
ns3-gym is a framework that integrates both OpenAI Gym and
ns-3 to encourage the usage of RL in networking research.

```bash
   git clone https://github.com/tkn-tub/ns3-gym.git <local-path>/ns-3.41/contrib/opengym
   cd <local-path>/ns-3.41/contrib/opengym
   git checkout app-ns-3.36+
```
* It is important to use the opengym as the name of the ns3-gym app directory.

### 1.4 Rebuild NS-3 with the additional modules

```bash
    cd <local-path>/ns-allinone-3.41/ns-3.41
    ./ns3 configure --enable-examples --enable-tests
    ./ns3 build
```
### 1.5 Install the ns3-gym module
* Opengym Protocol Buffer messages (C++ and Python) are built during configuration.
#### 1.5.1 Install the ns3-gym module
```bash
    cd <local-path>/ns-3.41/contrib/opengym
    pip3 install --user ./model/ns3gym
```
(Optional) Install all libraries required by your agent (like tensorflow, keras, etc.).

### 1.6 Install the A2C-LoVQI and DQN-LoVQI codes

Goes Get the code from the repository:
```bash
    cd <local-path>/ns-3.41/scratch
    git clone git@github.com:rogerio-silva/A2C-LoVQI.git
```

## 2. Run the code
Note that the Python code runs an agent and automatically starts the  ns-3 simulation. 
```bash
    cd <local-path>/ns-3.41/scratch/A2C-LoVQI
    python3 dqn_agent.py
    python3 a2c_agent.py
```
Enjoy it!

## 3. Cite this work
```bibtex
    @InProceedings{silvaRS2024, 
        author = {Silva, R. S. and Oliveira, R. R. and Carvalho, L. T. S. and Freitas, L. A. and Oliveira-Jr, A. C. and Cardoso, K. V. and Reis, C. B. and Xavier, P. S.},
        booktitle = {Anais do XLII Simpósio Brasileiro de Telecomunicações e Processamento de Sinais (SBrT)},
        title = {{Soluções baseadas em Aprendizado por reforço profundo para implantar VANTs como gateways LoRaWAN com foco na qualidade de serviço IOT}},
        month = {November},
        year = {2024},
        pages = {1--6},
        doi= {https://doi.org/10.14209/sbrt.2024.1571036460}
    }
```


