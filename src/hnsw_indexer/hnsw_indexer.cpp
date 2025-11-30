#include "datasets.h"
#include <iostream>
#include <string>
#include <omp.h>
#include <mpi.h>
#include <vector>
#include <set>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include <filesystem>
#include "hnswlib/hnswlib.h"
namespace fs = std::filesystem;

ABSL_FLAG(std::string, load, "", "");
ABSL_FLAG(int32_t, idx, 0, "");
ABSL_FLAG(int32_t, M, 20, "Maximum out-degree of the graph used in the index");
ABSL_FLAG(int32_t, efc, 200, "ef value used during index construction");

int main(int argc, char** argv){

    absl::ParseCommandLine(argc, argv);

    if(strcmp(absl::GetFlag(FLAGS_load).c_str(), "") == 0){
        printf("Require a index path. Use --load <path>\n");
        return 0;
    }

    hnswlib::HierarchicalNSW<float>* alg_hnsw;
    std::string prefix = absl::GetFlag(FLAGS_load);
    int idx = absl::GetFlag(FLAGS_idx);

    std::string hnsw_path = prefix + "hnsw_"+std::to_string(idx)+".bin";


    size_t nb, d_read;
    
    auto base_vector_path = (prefix + "pack_" + std::to_string(idx) + ".fvecs");
    float* xb = fvecs_read(base_vector_path.c_str(), &d_read, &nb);

    std::string id_file_path = (prefix + "id_" + std::to_string(idx) + ".bin");
    FILE* id_file = fopen(id_file_path.c_str(), "r");
    if (!id_file) {
        fprintf(stderr, "could not open %s\n", ("id_" + std::to_string(idx) + ".bin").c_str());
        perror("");
        abort();
    }

    int *ids = new int[nb];
    fread(ids, nb, sizeof(int), id_file);
    printf("number of vec: %d\n", nb);
    // for(int i=0;i<nb;++i){
    //     if(ids[i] < 0){
    //         printf("id %d is negative, %d-th vector\n", ids[i], i);
    //         fflush(stdout);
    //     }
    // }
    fclose(id_file);
    size_t d = d_read;
    hnswlib::L2Space space(d);
    alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, nb, absl::GetFlag(FLAGS_M), absl::GetFlag(FLAGS_efc));
    auto t0 = time_now();
    printf("graph use %d thread\n",  omp_get_max_threads());
    int percent = nb / 100;
    #pragma omp parallel for
    for (int i = 0; i < nb; i++) {
        //printf("%d\n", ids[i]);
        alg_hnsw->addPoint(xb + (long long)i * d, ids[i]);
        if (percent > 0 && i % percent == 0) {
            printf("add %d vec\n", i);
        }
    }
    delete[] xb;
    printf("time cost %lf\n", elapsed_s(t0));
    delete[] ids;
    
    std::cout<<"Add "<< nb << " vectors of " << d <<"-dim"<<std::endl;
    try{
        fs::remove(id_file_path);
        fs::remove(base_vector_path);
    }catch(const fs::filesystem_error& err){
        std::cout<<err.what()<<std::endl;
    }
    alg_hnsw->saveIndex(hnsw_path);
}