FROM borda/docker_python-opencv-ffmpeg:gpu-py3.10-cv4.6.0

# https://fabiorosado.dev/blog/install-conda-in-docker/
# https://github.com/sinzlab/pytorch-docker/blob/master/Dockerfile
# Install base utilities
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    pkg-config && \
    apt-get clean && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py &&\
    python3 get-pip.py &&\
    rm get-pip.py &&\
    # best practice to keep the Docker image lean
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


## Install miniconda
#ENV CONDA_DIR /opt/conda
#RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#     /bin/bash ~/miniconda.sh -b -p /opt/conda
#
## Put conda in path so we can use conda activate
#ENV PATH=$CONDA_DIR/bin:$PATH
#
#RUN conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
RUN pip3 install -U tqdm scipy Pillow


RUN pip3 install torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu116

#VOLUME /home/FootAndBall /home/FootAndBall/DATASETS
VOLUME /opt/project /DATASETS

#WORKDIR /home/FootAndBall
WORKDIR /opt/project
