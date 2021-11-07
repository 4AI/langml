FROM ubuntu:16.04

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH

RUN apt-get update && \
    apt-get install -y locales && \
    locale-gen en_US.UTF-8 && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends apt-utils \
    # dependency for scipy
    gfortran libatlas-dev libblas-dev libopenblas-dev liblapack-dev libopenblas-base \
    redis-server wget build-essential curl ca-certificates libssl-dev && \
    DEBIAN_FRONTEND=noninteractive apt-get autoremove -y && \
    rm -rf /var/lib/apt/list/*

# Install Python 3.7
RUN echo 'deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu xenial main' > /etc/apt/sources.list.d/deadsnakes-ubuntu-ppa-xenial.list && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 6A755776 && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3.7 python3.7-dev python3.7-venv && \
    curl -sSL https://bootstrap.pypa.io/get-pip.py | python3.7 \
    && python3.7 -m ensurepip

ADD dev-requirements.txt /tmp/dev-requirements.txt
ADD requirements.txt /tmp/requirements.txt

ARG PIP_INDEX_URL
RUN python3.7 -m pip install --no-cache-dir -U pip twine flake8 "tensorflow<2.5.0" && \
    python3.7 -m pip install --no-cache-dir -r /tmp/dev-requirements.txt
