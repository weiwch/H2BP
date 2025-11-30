# https://grpc.io/docs/languages/cpp/quickstart/

export MY_INSTALL_DIR=$HOME/.local
export PATH="$MY_INSTALL_DIR/bin:$PATH"

apt install -y build-essential autoconf libtool pkg-config
git clone --recurse-submodules -b v1.72.0 --depth 1 --shallow-submodules https://github.com/grpc/grpc

cd grpc
mkdir -p cmake/build
pushd cmake/build
cmake -DgRPC_INSTALL=ON \
      -DgRPC_BUILD_TESTS=OFF \
      -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_INSTALL_PREFIX=$MY_INSTALL_DIR \
      ../..
make -j$(nproc)
make install
popd