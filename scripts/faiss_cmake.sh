
# move to thirdparty/faiss

cmake -B build . -DFAISS_ENABLE_GPU=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=avx512 -DFAISS_USE_LTO=ON 

make -C build -j$(nproc) faiss faiss_avx2 faiss_avx512

cmake --install build --prefix _libfaiss_stage/