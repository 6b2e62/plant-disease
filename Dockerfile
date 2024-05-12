FROM ubuntu:22.04

# Packages
RUN apt-get update && apt-get upgrade && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \ 
    curl liblzma-dev python-tk python3-tk tk-dev libssl-dev libffi-dev libncurses5-dev zlib1g zlib1g-dev \
    libreadline-dev libbz2-dev libsqlite3-dev make gcc curl git-all wget python3-openssl gnupg2

# Setup CUDA
RUN apt-key del 7fa2af80 && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin && \
    mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda-repo-wsl-ubuntu-12-2-local_12.2.2-1_amd64.deb && \
    dpkg -i cuda-repo-wsl-ubuntu-12-2-local_12.2.2-1_amd64.deb && \
    cp /var/cuda-repo-wsl-ubuntu-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get -y install cuda-toolkit-12-2

# Pyenv
ENV PYENV_ROOT="$HOME/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PYENV_ROOT/versions/3.10.12/bin:$PATH"

RUN curl https://pyenv.run | bash
RUN pyenv install 3.10.12 && \
    pyenv global 3.10.12 && \
    echo 'eval "$(pyenv init --path)"' >> ~/.bashrc && \
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

SHELL ["/bin/bash", "-c"]

WORKDIR /app
ADD ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

ENV CUDNN_PATH="/.pyenv/versions/3.10.12/lib/python3.10/site-packages/nvidia/cudnn/"
ENV LD_LIBRARY_PATH="$CUDNN_PATH/lib":"/usr/local/cuda-12.2/lib64"
ENV PATH="$PATH":"/usr/local/cuda-12.2/bin"