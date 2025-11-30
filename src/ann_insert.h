#pragma once

#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include "partition/partition.h"
#include "utils.h"
typedef int64_t idx_t;

int ann_insert(PartitionAlgo* partition, const float* xq, size_t nq, size_t d, size_t* n_server, size_t nlist, int start_idx, int base_idx = 0){
    MPI_Request req[nq];
    idx_t* Iq = new idx_t[nq];
    {
        // omp_set_num_threads(1);
        partition->rank(nq, xq, 1, Iq);
    }
    float** buffer = new float*[nq];
    for (int i=0;i<nq;++i){
        
        int server = Iq[i] + 1;
        // printf("insert %d to %d\n", start_idx + i, server);
        // fflush(stdout);
        buffer[i] = new float[d+1];
        // memset(buffer[i], 0xff, sizeof(float)); // -1 means insert
        int idx = -(start_idx + i);
        *buffer[i] = *((float*)&(idx));
        memcpy(buffer[i]+1, xq + i * d, d * sizeof(float));
        MPI_Isend(buffer[i], d+1, MPI_FLOAT, server, i + base_idx, MPI_COMM_WORLD, &req[i]);
    }
    for(int i=0;i<nq;++i){
        MPI_Wait(&req[i], MPI_STATUS_IGNORE);
        delete[] buffer[i];
    }
    delete[] buffer;
    delete[] Iq;
    return 0;
}