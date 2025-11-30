#include <omp.h>
#include <mpi.h>
#include "hnswlib/hnswlib.h"
#include "utils.h"

template<bool record_source, bool support_insert = false>
void hnsw_mpi_server(MPI_Request *rq, int n_res_per_server, hnswlib::HierarchicalNSW<float>* alg_hnsw, float* vec, int rank){
    int byte_count = n_res_per_server*(sizeof(idx_t)+sizeof(float)) ;
    if constexpr(record_source){
        byte_count += sizeof(float);
    }
    char* buffer = new char[byte_count];
    float *pf = reinterpret_cast<float*>(buffer);
    idx_t *pi = reinterpret_cast<idx_t*>(buffer + n_res_per_server*sizeof(float));
    
    MPI_Status sts;
    MPI_Wait(rq, &sts);

    // printf("ef : %d\n", ef);
    //std::cout<<"recv "<<sts.MPI_TAG<<std::endl;
    // if(sts.MPI_TAG==1001){
    //     printf("rk %d ef %d\n", rank, *(int*)(vec));
    // }

    if constexpr(support_insert){
        int ef = *(int*)(vec);
        // printf("reach insert server 3 %d ef %d\n", rank, ef);
        if(ef<0){
            ef = -ef;
            printf("rank %d insert %d\n", rank, ef);
            fflush(stdout);
            alg_hnsw->addPoint(vec+1, ef);
            delete[] vec;
        }
        else{
            utils_time_point start_time;
            if constexpr(record_source){
                start_time = time_now();
            }
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(vec+1, n_res_per_server, nullptr, *(int*)(vec));
            delete[] vec;
            while (!result.empty())
            {
                auto &tmp = result.top();
                *pf++ = tmp.first;
                *pi++ = tmp.second;
                //printf("%f %ld \n",tmp.first, tmp.second);
                result.pop();
            }
            if constexpr(record_source){
                
                float search_time = elapsed_ms(start_time);
                float *plast = reinterpret_cast<float*>(buffer + byte_count - sizeof(float));
                *plast = search_time;
            }
            MPI_Request rq_send;
            MPI_Isend(buffer, byte_count, MPI_BYTE, 0, sts.MPI_TAG, MPI_COMM_WORLD, &rq_send);

            MPI_Wait(&rq_send, MPI_STATUS_IGNORE);
        }

    }else{
        
        utils_time_point start_time;
        if constexpr(record_source){
            start_time = time_now();
        }
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(vec+1, n_res_per_server, nullptr, *(int*)(vec));
        delete[] vec;
        while (!result.empty())
        {
            auto &tmp = result.top();
            *pf++ = tmp.first;
            *pi++ = tmp.second;
            //printf("%f %ld \n",tmp.first, tmp.second);
            result.pop();
        }
        if constexpr(record_source){
            
            float search_time = elapsed_ms(start_time);
            float *plast = reinterpret_cast<float*>(buffer + byte_count - sizeof(float));
            *plast = search_time;
        }
        MPI_Request rq_send;
        MPI_Isend(buffer, byte_count, MPI_BYTE, 0, sts.MPI_TAG, MPI_COMM_WORLD, &rq_send);

        MPI_Wait(&rq_send, MPI_STATUS_IGNORE);
    }
    delete[] buffer;
}