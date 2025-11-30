#include "datasets.h"
#include "partition/partition.h"
#include "partition/partition_kmeans.h"
#include "partition/partition_hybrid.h"
#include "partition/partition_graph.h"
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

ABSL_FLAG(int32_t, M, 20, "Maximum out-degree of the graph used in the index");
ABSL_FLAG(int32_t, efc, 200, "ef value used during index construction");
ABSL_FLAG(bool, seq, false, "Sequential building");

int main(int argc, char** argv){
    absl::ParseCommandLine(argc, argv);

    if(strcmp(absl::GetFlag(FLAGS_load).c_str(), "") == 0){
        printf("Require a index path. Use --load <path>\n");
        return 0;
    }


    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        std::cerr << "MPI does not support MPI_THREAD_MULTIPLE, provided: " << provided << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::string prefix = absl::GetFlag(FLAGS_load) + "/";
    
    std::cout<<"start hnsw "<<rank<<std::endl;
    hnswlib::HierarchicalNSW<float>* alg_hnsw;
    
    int nlist = world_size;
    int idx = rank%nlist;

    std::string hnsw_path = prefix + "hnsw_"+std::to_string(idx)+".bin";

    // wait for previous rank to finish
    if(absl::GetFlag(FLAGS_seq) && rank > 0){
        int msg;
        MPI_Recv(&msg, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    std::cout<<"pass hnsw "<<rank<<std::endl;

    size_t nb, d_read;
    
    auto base_vector_path = (prefix + "pack_" + std::to_string(idx) + ".fvecs");
    float* xb = fvecs_read(base_vector_path.c_str(), &d_read, &nb);
    // load from disk while building graph
    std::string id_file_path = (prefix + "id_" + std::to_string(idx) + ".bin");
    FILE* id_file = fopen(id_file_path.c_str(), "r");
    if (!id_file) {
        fprintf(stderr, "could not open %s\n", ("id_" + std::to_string(idx) + ".bin").c_str());
        perror("");
        abort();
    }

    int *ids = new int[nb];
    fread(ids, nb, sizeof(int), id_file);
    fclose(id_file);
    size_t d = d_read;
    hnswlib::L2Space space(d);
    alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, nb<<1, absl::GetFlag(FLAGS_M), absl::GetFlag(FLAGS_efc));
    auto t0 = time_now();
    printf("build %d graph %d use (omp thread): %d\n", rank, idx, omp_get_max_threads());
    #pragma omp parallel for // num_threads(omp_get_max_threads()/(world_size-1)*2)
    for (int i = 0; i < nb; i++) {
        //printf("%d\n", ids[i]);
        alg_hnsw->addPoint(xb + (long long)i * d, ids[i]);
    }
    delete[] xb;
    printf("build %d time cost %lf\n", rank, elapsed_s(t0));
    delete[] ids;
    
    std::cout<<"Add "<< nb << " vectors of " << d <<"-dim"<<std::endl;
    try{
        fs::remove(id_file_path);
        fs::remove(base_vector_path);
    }catch(const fs::filesystem_error& err){
        std::cout<<err.what()<<std::endl;
    }
    alg_hnsw->saveIndex(hnsw_path);
    delete alg_hnsw;
    // pass message to next rank to make it start
    if(absl::GetFlag(FLAGS_seq) && rank < world_size - 1){
        int msg = 1;
        MPI_Send(&msg, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}
