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

template<bool record_source>
void ann_search(const float* xq, size_t nq, size_t d, idx_t* res, size_t n_result, size_t n_res_per_server, size_t* n_server, size_t nlist, size_t nsearch, int* ief, PartitionAlgo* partition, size_t* from_od = nullptr, double* cost = nullptr, int base_idx = 0){    
    int nns = n_res_per_server;
    int byte_pre_server = nns*(sizeof(idx_t)+sizeof(float));

    if constexpr(record_source){
        byte_pre_server += sizeof(float);
    }

    int byte_count = nsearch*byte_pre_server;
    MPI_Request rq_recv[nq][nsearch];
    MPI_Request req[nq][nsearch];
    
    char** buffers = new char*[nq];
    float* tmpq = new float[nq*nsearch*(d+1)];
    idx_t* Iq = new idx_t[nsearch * nq];
    if(nsearch <= nlist){
        // float* Dq = new float[nsearch * nq];
        // printf("start rank\n");
        // fflush(stdout);
        auto st = time_now();
        {
            // omp_set_num_threads(1);
            partition->rank(nq, xq, nsearch, Iq);
        }
        double rank_time = elapsed_ms(st);
        printf("rank time: %lf ms\n", rank_time);
        for (int i=0;i<nq;++i){
            
            idx_t* need_search_ids = Iq + i *nsearch;

            buffers[i] = new char[byte_count];

            for(int t=0;t<nsearch/*nsearch*/;++t){
                int s = need_search_ids[t];
                if (s < 0 || s >= nlist) {
                    // printf("ERROR: %d %d %d\n", s, nlist, i);
                    continue; // skip invalid server
                }
                int server = s + (i%n_server[s])*nlist +1;
                // tmpq[d] = *(float*)&ef;

                //tmpq[i*nsearch+t] = new float[d+1];
                float *p = tmpq + i*(nsearch*(d+1)) + t *(d+1);
                
                *p = *((float*)&(ief[t]));
                memcpy(p+1, xq + i * d, d * sizeof(float));
                // int ief = ef * pow(0.5,t);
                
                // p[d] = ief[t];
                //memcpy(p , &ief[t], sizeof(float));
                // float *p =  xq + i * d;
                if (false && i<100){
                    // for(int i=0;i<d;++i){
                    //     //if(tmpq[i*nsearch+t][i]!=(xq + i * d)[i]){
                    //         printf("%f %f\n", tmpq[i*nsearch+t][i], p[i]);
                    //     //}
                    // }
                    //printf("%d: %d send to %d(rk %d)\n", t, *((int*)(p+d)), s, server);
                    printf("%d: %d send to %d(rk %d)\n", t, *(int*)(p), s, server);

                }
                // printf("OKOKOKOK BEFORE SENT\n");
                // fflush(stdout);
                
                MPI_Isend(p, d + 1, MPI_FLOAT, server, i+1 + base_idx, MPI_COMM_WORLD, &req[i][t]); //  &req[t]

                MPI_Irecv(buffers[i] + t *(byte_pre_server), byte_pre_server, MPI_BYTE, server, i+1 + base_idx, MPI_COMM_WORLD, &rq_recv[i][t]);

                // ++now_server[s];
                // now_server[s] %= n_server[s];
            }
        }


        
        // delete[] Dq;
    }

    // printf("OKOKOKOK IN MIDDLE");
    // fflush(stdout);
    for (int i=0;i<nq;++i){
        if constexpr(record_source){
            std::vector<std::tuple<float, idx_t, idx_t> > all_res;
            for(int t=0;t<nsearch/*nsearch*/;++t){
                //printf("OKOKOKOK BEFORE RECV\n");fflush(stdout);
                idx_t* need_search_ids = Iq + i *nsearch;
                int s = need_search_ids[t];
                if (s < 0 || s >= nlist) {
                    // printf("ERROR: %d %d %d\n", s, nlist, i);
                    continue; // skip invalid server
                }
                MPI_Wait(&req[i][t], MPI_STATUS_IGNORE);
                MPI_Wait(&rq_recv[i][t], MPI_STATUS_IGNORE);

                char *tbuffer = buffers[i] + t *(byte_pre_server);
                float* distances = reinterpret_cast<float*>(tbuffer);
                idx_t* ids = reinterpret_cast<idx_t*>(tbuffer + nns*sizeof(float));
                float *plast = reinterpret_cast<float*>(tbuffer + byte_pre_server - sizeof(float));
                cost[t] += *plast;
                for(int j=0;j<nns;++j){
                    all_res.push_back(std::make_tuple(distances[j], ids[j], t));
                }
                // delete[] tmpq[i*nsearch+t];// BTW

            }
            delete[] buffers[i];
            std::sort(all_res.begin(),all_res.end());
            for(int t = 0;t<n_result;++t){
                auto pos = i*n_result + t;
                res[pos] = std::get<1>(all_res[t]);
                from_od[pos] = std::get<2>(all_res[t]);
            }
        }else{
            std::vector<std::pair<float, idx_t> > all_res;  
            for(int t=0;t<nsearch/*nsearch*/;++t){
                idx_t* need_search_ids = Iq + i *nsearch;
                int s = need_search_ids[t];
                if (s < 0 || s >= nlist) {
                    // printf("ERROR: %d %d %d\n", s, nlist, i);
                    continue; // skip invalid server
                }
                MPI_Wait(&req[i][t], MPI_STATUS_IGNORE);
                MPI_Wait(&rq_recv[i][t], MPI_STATUS_IGNORE);
                char *tbuffer = buffers[i] + t *(byte_pre_server);
                float* distances = reinterpret_cast<float*>(tbuffer);
                idx_t* ids = reinterpret_cast<idx_t*>(tbuffer + nns*sizeof(float));
                for(int j=0;j<nns;++j){
                    all_res.push_back(std::make_pair(distances[j], ids[j]));
                }
                // delete[] tmpq[i*nsearch+t];// BTW

            }
            delete[] buffers[i]; // for use behind
            std::sort(all_res.begin(),all_res.end());
            for(int t = 0;t<n_result;++t){
                res[i*n_result + t] = all_res[t].second;
            }
        }
    }
    delete[] Iq;
    delete[] buffers;
    delete[] tmpq;
}
