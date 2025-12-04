FROM ubuntu:latest

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 安装系统依赖和 Python 开发环境
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        gcc \
        g++ \
        make \
        pkg-config \
        libffi-dev \
        zlib1g-dev \
        libssl-dev \
        libcairo2-dev \
        libxinerama-dev \
        libxext-dev \
        libxrender-dev \
        libxi-dev \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
        tzdata && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        python3.10-distutils && \
    ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/include/python3.10 /usr/include/python3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 创建虚拟环境
RUN python3.10 -m venv $VIRTUAL_ENV

# 设置 Python 包含路径环境变量
ENV C_INCLUDE_PATH="/usr/include/python3.10:$C_INCLUDE_PATH"
ENV CPLUS_INCLUDE_PATH="/usr/include/python3.10:$CPLUS_INCLUDE_PATH"

# copy requirements
COPY ./backend/llm-webkit-mirror/requirements/dev.txt ./backend/llm-webkit-mirror/requirements/runtime.txt ./backend/llm-webkit-mirror/requirements/requirements.txt ./

# 安装 pip 依赖
RUN pip install --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple/ && \
    # 先尝试安装可能引起问题的包
    pip install wheel setuptools_scm -i https://pypi.tuna.tsinghua.edu.cn/simple/ && \
    # 然后安装所有依赖
    pip install -r dev.txt -r runtime.txt -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

COPY ./models/checkpoint-3296/ /root/checkpoint-3296

COPY ./backend/llm-webkit-mirror/ .

ENV MODEL_PATH=/root/checkpoint-3296

CMD ["python", "-m", "uvicorn", "llm_web_kit.api.main:app", "--host", "0.0.0.0", "--port", "9501"]