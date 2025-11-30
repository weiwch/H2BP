#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include "datasets.h"
#include <omp.h>
#include <mpi.h>
#include "hnswlib/hnswlib.h"
#include <string>
#include <iostream>
#include "utils.h"

#include "ann_search.h"
#include "ann_mpi_service.h"
#include <set>

#include "partition/partition.h"
#include "partition/partition_kmeans.h"
#include "partition/partition_hybrid.h"
#include "partition/partition_graph.h"
#include "partition/partition_loader.h"
#include "config.h"
using faiss::idx_t;

ABSL_FLAG(std::string, load, "", "");
// ABSL_FLAG(std::string, query, "../../deep/query.public.10K.fbin", "");
// ABSL_FLAG(std::string, gt, "../../deep/deep100M_groundtruth.ivecs", "");
ABSL_FLAG(std::string, query, "../../sift1M/sift_query.fvecs", "");
ABSL_FLAG(std::string, gt, "../../sift1M/sift_groundtruth.ivecs", "");
ABSL_FLAG(int32_t, check_k, 10, "# of checked nnns per query");

ABSL_FLAG(int32_t, smpl, -1000, "gen a table use sample size query points");
ABSL_FLAG(int32_t, start, 50, "");
ABSL_FLAG(int32_t, stop, 1000, "");
ABSL_FLAG(int32_t, interval, 50, "");

const bool record_time_src = false;

int main(int argc, char** argv){

    absl::ParseCommandLine(argc, argv);

    if(strcmp(absl::GetFlag(FLAGS_load).c_str(), "") == 0){
        printf("Require a index path. Use --load <path>\n");
        return 0;
    }

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    if(rank != 0){
        
        const std::string filePath = absl::GetFlag(FLAGS_load) + "/conf.json";
        auto cfg = Config(filePath);
        hnswlib::HierarchicalNSW<float>* alg_hnsw;
        int d = cfg.get<int>("dim"); 
        hnswlib::L2Space space(d);
        int nlist = cfg.get<int>("nlist");
        int gid = (rank-1)%nlist;
        std::string prefix = absl::GetFlag(FLAGS_load) + "/";
        std::string hnsw_path = prefix + "hnsw_"+std::to_string(gid)+".bin";
        std::cout<<rank<<": read from file "<<hnsw_path<<std::endl;
        alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);
        {
        // auto h = alg_hnsw;
        // int n = h->max_elements_;
        // int lim = 10;
        // int cnt = 0;
        // for(int i=0;i<n;++i){
        //     int *data = (int *) h->get_linklist0(i);
        //     size_t size = h->getListCount((unsigned int*)data);
        //     if(size > 2*lim){
        //         h->setListCount((unsigned int*)data, 2*lim);
        //         ++cnt;
        //     }
        // }
        // printf("%d : out of size: %d", rank, cnt);
        }
        std::cout<<rank<<": read from file OK "<<hnsw_path<<std::endl;
        
        int nns = absl::GetFlag(FLAGS_check_k); // # of nearest neighbor each server
        int &n_res_per_server = nns;
        printf("rank %d graph %d use %d thread\n", rank, gid, omp_get_max_threads());
        MPI_Barrier(MPI_COMM_WORLD); // MPI_Barrier2 : prepare OK, start to search
        #pragma omp parallel // num_threads(omp_get_max_threads()/(world_size-1))
        {
            while (true){
                float *vec = new float[d+1];
                MPI_Request req_v;
                MPI_Irecv(vec, d+1, MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &req_v);
                // printf("%d ef\n",*(int*)(vec+d));
                if(absl::GetFlag(FLAGS_smpl)>0){
                    hnsw_mpi_server<true>(&req_v, n_res_per_server, alg_hnsw, vec, rank);
                }
                else{
                    hnsw_mpi_server<record_time_src>(&req_v, n_res_per_server, alg_hnsw, vec, rank);
                }
                //delete[] vec;
            }
        }
    }
    else{
        Dataset ds("", absl::GetFlag(FLAGS_query), absl::GetFlag(FLAGS_gt));
        ds.read_query();

        
        std::string prefix = absl::GetFlag(FLAGS_load) + "/";
        std::shared_ptr<PartitionAlgo> partition = load_from_file(prefix);
        int nlist = partition->nlist;
        // faiss::IndexFlat* pquantizer = &partition->quantizer;
        size_t n_server[nlist];
        
        printf("nlist:%d world_size:%d\n", nlist, world_size);
        int avg = (world_size-1)/nlist;
        for(int t=0;t<nlist;++t){
            n_server[t] = avg;
            //now_server[t] = 0;
        }
        for(int t=0;t<(world_size-1)%nlist;++t){
            ++n_server[t];
        }
        for(int t=0;t<nlist;++t){
            printf("n_server[%d] = %d\n", t, n_server[t]);
        }

        int nnns = absl::GetFlag(FLAGS_check_k); // # of nearest neighors
        int &n_res_per_server = nnns;
        idx_t* Ir = new idx_t[nnns * ds.nq];
        
        MPI_Barrier(MPI_COMM_WORLD); // MPI_Barrier2 

        if(absl::GetFlag(FLAGS_smpl)>0){
            int sample_size = absl::GetFlag(FLAGS_smpl);
            int stop = absl::GetFlag(FLAGS_stop);
            int interval = absl::GetFlag(FLAGS_interval);
            std::string filename = prefix + "all_res.csv";
            FILE* fp = fopen(filename.c_str(), "w");
            bool warmup = true;
            for(int ef = absl::GetFlag(FLAGS_start); ef <= stop; ef += interval){
                int ief[nlist];
                for(int i=0;i<nlist;++i){
                    ief[i] = ef ;//* pow(0.5, i) ;//* pow(0.3, i);
                }
                auto st = time_now();
                size_t* from_od = new size_t[nnns * ds.nq];
                int nsearch = nlist; //search all
                double* cost = new double[nsearch];
                memset(cost, 0, sizeof(double)*nsearch);
                ann_search<true>(ds.xq, sample_size, ds.d, Ir, nnns, n_res_per_server, n_server, nlist, nsearch, ief, partition.get(), from_od, cost);
                double search_time = elapsed_ms(st);
                std::cout<<search_time<<" ms"<<std::endl;
                std::cout<<search_time/sample_size<<" ms (per query)"<<std::endl;
                int shit = 0;
                std::vector<int> count_od(nlist);
                for(int i=0;i<sample_size;++i){
                    std::set<idx_t> st;
                    int hit = 0;
                    for(int j=0;j<nnns;++j){
                        st.insert(ds.gt[i*ds.gt_k+j]);
                    }
                    for(int j=0;j<nnns;++j){
                        if(st.count(Ir[i*nnns+j])){
                            hit++;
                            auto frm = from_od[i*nnns+j];
                            count_od[frm]++;
                        }
                    }
                    shit+=hit;
                }
                printf("recall@%d : %d %.5f\n",nnns,shit,1.0*shit/(sample_size*nnns));
                
                if(warmup){
                    warmup = false;
                    ef = absl::GetFlag(FLAGS_start) - interval;
                    continue;
                }
                // int suffix = 0;
                // std::string new_filename = filename;
                // while (access(new_filename.c_str(), F_OK) != -1) {
                //     suffix++;
                //     new_filename = filename + "(" + std::to_string(suffix)+ ")";
                // }
                // FILE* fp = fopen(new_filename.c_str(), "a");
                // fprintf(fp, "%lf %lf", 1.0*shit/(nq*nns), search_time/nq);
                for(int i=0;i<nlist;++i){
                    printf("recall@%d : %d-th server: %.5f, cost %.5lf, ef %d\n",nnns,i,1.0*count_od[i]/(sample_size*nnns), cost[i]/sample_size, ief[i]);
                    fprintf(fp, "%d %d %lf %lf\n", i, ief[i], 1.0*count_od[i]/(sample_size*nnns), cost[i]/sample_size);
                }
            }
            fclose(fp);
        }
        else{
            int test_offset = - absl::GetFlag(FLAGS_smpl);
            std::string filename = prefix + "ef_file.csv";
            FILE* fp = fopen(filename.c_str(), "r");
            if (!fp) {
                fprintf(stderr, "could not open %s\n", filename.c_str());
                perror("");
                abort();
            }
            std::string filename_res = prefix + "res.csv";
            FILE* fp_res = fopen(filename_res.c_str(), "w");
            int nsearch = 0; // init
            bool warmup = true;
            for(int i=0;;++i){
                int ief[nlist];
                double exp_tgt;
                int read = 0;
                for(int j=0;j<nlist;++j){
                    read = fscanf(fp, "%d", ief+j);
                    if(read == EOF){
                        break;
                    }
                    if(ief[j] != -1){
                        nsearch = j+1; // actual # of searched servers 
                    }
                    printf("%d ", ief[j]);
                }
                if(read == EOF){
                    break;
                }
                printf(" result %d-th:\n----------------------\n", i);
                fscanf(fp, "%lf", &exp_tgt);
            warmup_loop:
                printf("search %d posting lists now\n",nsearch);
                auto st = time_now();
                size_t* from_od = nullptr;
                double* cost = nullptr;
                if(record_time_src){
                    from_od = new size_t[nnns * ds.nq];
                    cost = new double[nsearch];
                    memset(cost, 0, sizeof(double)*nsearch);
                }
                ann_search<record_time_src>(ds.xq + test_offset * ds.d, ds.nq - test_offset, ds.d, Ir, nnns, nnns, n_server, nlist, nsearch, ief, partition.get(), from_od, cost);

                double search_time = elapsed_ms(st);
                int real_nq = ds.nq - test_offset;
                std::cout<<search_time<<" ms"<<std::endl;
                std::cout<<search_time/real_nq<<" ms (per query)"<<std::endl;
                
                if(warmup){
                    warmup = false;
                    goto warmup_loop;
                }

                int shit = 0;
                for(int i = test_offset; i < ds.nq ;++i){
                    // if(false && i<100)
                    // printf("%d:\n",i);
                    std::set<idx_t> st;
                    int hit = 0;
                    for(int j=0;j<nnns;++j){
                        // if(i<100)
                        //     printf("%d ", ds.gt[i*ds.gt_k+j]);
                        st.insert(ds.gt[i*ds.gt_k+j]);
                    }
                    for(int j=0;j<nnns;++j){
                        // if(i<100)
                        //     printf("%d ", Ir[i*nnns+j]);
                        if(st.count(Ir[(i-test_offset)*nnns+j])){
                            hit++;
                        }
                    }
                    shit+=hit;
                }
                printf("recall@%d : %d %.5f\n",nnns,shit,1.0*shit/(real_nq*nnns));
                fprintf(fp_res, "%lf\t%lf\n", 1.0*shit/(real_nq*nnns), search_time/real_nq);
            }
            fclose(fp);
            fclose(fp_res);
        }
        delete[] Ir;
    }
    MPI_Abort(MPI_COMM_WORLD, 0);
    // MPI_Finalize();
    return 0;
}