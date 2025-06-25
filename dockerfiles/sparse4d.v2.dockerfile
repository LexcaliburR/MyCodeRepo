FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 换apt源为阿里云
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y wget bzip2 git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 安装Miniconda
RUN wget https://mirrors.aliyun.com/miniconda/Miniconda3-py39_23.11.0-1-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# 换conda源为阿里云
RUN conda config --add channels https://mirrors.aliyun.com/pypi/simple/ && \
    conda config --add channels https://mirrors.aliyun.com/anaconda/pkgs/main/ && \
    conda config --add channels https://mirrors.aliyun.com/anaconda/pkgs/free/ && \
    conda config --set show_channel_urls yes

# 换pip源为阿里云
RUN mkdir -p /root/.pip && \
    echo "[global]\nindex-url = https://mirrors.aliyun.com/pypi/simple/" > /root/.pip/pip.conf

# 创建sparse4D环境并安装依赖
RUN conda create -n sparse4D python=3.9 -y && \
    /bin/bash -c "source activate sparse4D && \
    conda install -y numba=0.53.1 numpy=1.19.5 && \
    pip install nuscenes-devkit==1.1.9 \
        mmcv==1.4.8 \
        mmcv-full==1.4.0 \
        mmdet==2.19.1 \
        mmdet3d==1.0.0rc0 \
        mmsegmentation==0.20.2 \
        motmetrics==1.1.3 \
        tensorboard==2.6.0 \
        torch==1.9.1+cu111 \
        torchmetrics==0.5.0 \
        torchvision==0.10.1+cu111 \
    && conda clean -afy && pip cache purge"

# 默认进入sparse4D环境
SHELL ["conda", "run", "-n", "sparse4D", "/bin/bash", "-c"]

CMD ["/bin/bash"]