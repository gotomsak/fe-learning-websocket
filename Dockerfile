FROM python:3.6-slim

WORKDIR /app
COPY requirements.txt /app/

ENV LIB="libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libavformat-dev \
    libpq-dev"

RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get install --no-install-recommends -qy $LIB \
    && apt-get clean \
    && apt-get autoclean \
    && apt-get autoremove \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /tmp/* /var/tmp/* \
    && rm -rf /var/lib/apt/lists/* \
    rm -rf /var/lib/apt/lists/*

CMD [ "python", "./server.py" ]