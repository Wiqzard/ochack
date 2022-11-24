FROM nvidia/cuda:11.2.0-devel-ubuntu18.04

#ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update  -y --fix-missing && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    curl \
    unrar \
    unzip \
    git && \
    apt-get upgrade -y libstdc++6 && \
    apt-get clean -y

RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y gcc-9 && \
    apt-get upgrade -y libstdc++6

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b  && \
    rm -rf Miniconda3-latest-Linux-x86_64.sh

RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
    --path-update=false --bash-completion=false \
    --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

RUN git clone https://github.com/Wiqzard/ochack.git /root/src

ENV PATH=/miniconda/bin:${PATH}

RUN conda update -y conda

RUN conda install -n base -c conda-forge mamba

ADD ./environment.yml ./environment.yml

RUN mamba env update -n base -f ./environment.yml && \
    conda clean -afy


WORKDIR /root

