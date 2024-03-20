FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV NVIDIA_VISIBLE_DEVICES all

RUN ln -sf /usr/share/zoneinfo/Asia/Tashkent /etc/localtime
RUN echo "Asia/Tashkent" > /etc/timezone

ENV py_version=3.9.16

RUN apt update -y
RUN apt install build-essential gcc autoconf make automake \
    wget libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
    libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev libffi-dev uuid-dev  -y \
    ffmpeg libpq-dev libsm6 libxext6 python3-pip

RUN wget https://www.python.org/ftp/python/${py_version}/Python-${py_version}.tgz
RUN tar xzf Python-${py_version}.tgz

WORKDIR Python-${py_version}

RUN ./configure --enable-optimizations

RUN make altinstall
RUN export PATH=/opt/python-${py_version}/bin:$PATH

RUN ln -sf /usr/local/bin/python3.9 /usr/bin/python3 

WORKDIR /app

COPY requirements.txt .

RUN python3 -m pip --no-cache-dir install -U pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt
    
RUN python3 -m pip install onnxruntime-gpu==1.14.1

COPY . .


CMD ["python3", "run.py"]
