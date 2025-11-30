FROM ubuntu:noble

# ARG REQUIRE="sudo build-base"
SHELL ["/bin/bash", "-c"]

RUN apt update \
    && apt install -y build-essential openssh-client mpich wget openssh-server git autoconf libtool pkg-config vim cmake htop 

# COPY ../cmake-3.24.4-linux-x86_64.sh  ./
# RUN cd ~ && wget https://github.com/Kitware/CMake/releases/download/v3.27.9/cmake-3.27.9-linux-x86_64.sh \
#     && bash cmake-3.27.9-linux-x86_64.sh --prefix=/usr/local --skip-license && rm cmake-3.27.9-linux-x86_64.sh 

# RUN echo 'export PATH=$PATH:/usr/local/bin' >> ~/.bashrc

# for anns

# install mkl

RUN apt install -y gpg-agent wget \
    && wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list \
    && apt update \
    && apt install -y intel-oneapi-hpc-toolkit \
    && echo 'source /opt/intel/oneapi/setvars.sh > /dev/null' >> ~/.bashrc

# # install faiss
# COPY ./thirdparty/faiss faiss

# RUN source /opt/intel/oneapi/setvars.sh > /dev/null \
#     && cd faiss \
#     && cmake -B _build \
#         -DBUILD_SHARED_LIBS=ON \
#         -DBUILD_TESTING=OFF \
#         -DFAISS_OPT_LEVEL=avx2 \
#         -DFAISS_ENABLE_GPU=OFF \
#         -DFAISS_ENABLE_PYTHON=OFF \
#         -DBLA_VENDOR=Intel10_64lp \
#         -DCMAKE_INSTALL_LIBDIR=lib \
#         -DCMAKE_BUILD_TYPE=Release .

# RUN cd faiss && cd _build && make -j install

RUN git clone --recurse-submodules -b v1.72.0 --depth 1 --shallow-submodules https://github.com/grpc/grpc
# COPY ./thirdparty/grpc grpc

RUN export MY_INSTALL_DIR=$HOME/.local \
    && export PATH="$MY_INSTALL_DIR/bin:$PATH" \
    && cd grpc && mkdir -p cmake/build \
    && cd cmake/build \
    && cmake -DgRPC_INSTALL=ON \
        -DgRPC_BUILD_TESTS=OFF \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_INSTALL_PREFIX=$MY_INSTALL_DIR \
        ../.. \
    && make -j$(nproc) && make install && ldconfig

COPY scripts/entrypoint.sh ./
