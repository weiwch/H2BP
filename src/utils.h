#pragma once

#include <cstddef>
#include <string>
#include <cassert>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <ctime>
#include <chrono>
#include <string>
#include <sys/time.h>
#include "absl/time/time.h"
#include "absl/time/clock.h"
#include "absl/log/check.h"

inline std::chrono::high_resolution_clock::time_point time_now(){
   return std::chrono::high_resolution_clock::now();
}

inline double elapsed_ms(std::chrono::high_resolution_clock::time_point from){
    return std::chrono::duration_cast<std::chrono::nanoseconds>(time_now() - from).count() / 1000000.0;
}

inline double elapsed_s(std::chrono::high_resolution_clock::time_point from){
    return std::chrono::duration_cast<std::chrono::nanoseconds>(time_now() - from).count() / 1000000000.0;
}

typedef  std::chrono::high_resolution_clock::time_point utils_time_point;

inline double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

inline std::string getCurrentTimeString() {
    absl::Time now = absl::Now();
    absl::TimeZone local_time_zone;
    absl::LoadTimeZone("localtime", &local_time_zone);
    return absl::FormatTime("%y%m%d_%H%M%S", now, local_time_zone);
}

inline float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    CHECK(d >= 0 && d < 1000000) << "unreasonable dimension";
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    CHECK(sz % ((d + 1) * 4) == 0) << "weird file size";
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    CHECK(nr == n * (d + 1)) << "could not read whole file";

    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

inline int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

inline void fvecs_write(const char* fname, size_t d_out, size_t n_out, float* data) {
    FILE* f = fopen(fname, "wb");
    for (size_t i = 0; i < n_out; ++i) {
        fwrite(&d_out, sizeof(float), 1, f);
        fwrite(data + d_out * i, sizeof(float), d_out, f);
    }
    fclose(f);
}

inline void ivecs_write(const char* fname, size_t d_out, size_t n_out, int* data) {
    fvecs_write(fname, d_out, n_out, (float*)data);
}

inline float* fbin_read(const char* filename, size_t* d_out, size_t* n_out, size_t start_idx = 0, size_t chunk_size = 0) {
    FILE* file = fopen(filename, "rb");
    CHECK(file != NULL) << "Failed to open file";
    int nvecs, dim;
    auto ret = fread(&nvecs, sizeof(int), 1, file);
    CHECK(ret == 1) << "Failed to read number of vectors";
    ret = fread(&dim, sizeof(int), 1, file);
    CHECK(ret == 1) << "Failed to read dimension";

    CHECK((dim > 0 && dim < 1000000)) << "Unreasonable dimension";

    size_t nvecs_to_read = (chunk_size == 0) ? (size_t)nvecs - start_idx : chunk_size;
    CHECK(start_idx < (size_t)nvecs) <<  "Start index out of range";
    CHECK(chunk_size <= (size_t)nvecs - start_idx ) << "Chunk size out of range";

    float* data = new float[nvecs_to_read * dim];
    CHECK(data != NULL ) << "Failed to allocate memory";
    ret = fseek(file, start_idx * dim * sizeof(float) + 2 * sizeof(int), SEEK_SET);
    CHECK(ret == 0 ) << "Failed to seek to correct position";
    ret = fread(data, sizeof(float), nvecs_to_read * dim, file);
    CHECK(ret == nvecs_to_read * dim) << "Failed to read float32 vectors";
    
    *d_out = dim;
    *n_out = nvecs_to_read;

    fclose(file);
    return data;
}

inline int* ibin_read(const char* filename, size_t* d_out, size_t* n_out, size_t start_idx = 0, size_t chunk_size = 0) {
    return (int*)fbin_read(filename, d_out, n_out, start_idx, chunk_size);
}

inline void fbin_write(const char* filename, size_t d, size_t n, const float* data) {
    FILE* file = fopen(filename, "wb");
    CHECK(file != NULL) << "Failed to open file for writing";
    int nvecs = (int)n;
    int dim = (int)d;
    fwrite(&nvecs, sizeof(int), 1, file);
    fwrite(&dim, sizeof(int), 1, file);
    
    printf("writing %d vectors of dimension %d to %s\n", nvecs, dim, filename);
    fwrite(data, sizeof(float), n * d, file);
    fclose(file);
}

inline void ibin_write(const char* filename, size_t d, size_t n, const int* data) {
    fbin_write(filename, d, n, (const float*)data);
}
