#include <faiss/IndexFlat.h>
#include "utils.h"
#include "ann_search.h"
#include <faiss/IndexIVFFlat.h>
#include <faiss/Index.h>
#include <iostream>
#include <memory>
#include <string>
#include <omp.h>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include <vector>
#include "hnswlib/hnswlib.h"
#include <mpi.h>
#include <set>
#include <faiss/index_io.h>
#include <grpc/support/log.h>
#include "grpc_server.h"
#include "grpc_async_call.h"
#include "absl/log/check.h"
#include <absl/strings/str_format.h>
#include "ann_mpi_service.h"
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "datasets.h"

using namespace rapidjson;
typedef int64_t idx_t;
using absl::GetFlag;

GrpcAsyncServer::GrpcAsyncServer(const std::string &working_dir, std::shared_ptr<PartitionAlgo> parti_algo)
{
    working_dir_ = working_dir;
    parti_algo_ = parti_algo;
}

GrpcAsyncServer::~GrpcAsyncServer()
{
    server_->Shutdown();
    // Always shutdown the completion queue after the server.
    cq_->Shutdown();
}

void GrpcAsyncServer::Run(const uint16_t &port, const int num_threads)
{
    std::string server_address = absl::StrFormat("0.0.0.0:%d", port);

    ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    // Register "service_" as the instance through which we'll communicate with
    // clients. In this case it corresponds to an *asynchronous* service.
    builder.RegisterService(&service_);
    // Get hold of the completion queue used for the asynchronous communication
    // with the gRPC runtime.
    cq_ = builder.AddCompletionQueue();
    // Finally assemble the server.
    server_ = builder.BuildAndStart();
    std::cout << "Server listening on " << server_address << std::endl;

    // Proceed to the server's main loop.
    // HandleRpcs();
    //const int num_threads = std::thread::hardware_concurrency();
    printf("Use %d threads to handle grpc requests\n", num_threads);
    fflush(stdout);
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([this]() {
            HandleRpcs(); 
        });
    }
    for (auto &thread : threads) {
        thread.join();
    }
}

void GrpcAsyncServer::HandleRpcs()
{
    // Spawn a new CallData instance to serve new clients.
    CallData data{&service_, cq_.get(), parti_algo_};

    new SearchCall(&data);
    new BatchSearchCall(&data);
    new InsertCall(&data);

    void* tag;  // uniquely identifies a request.
    bool ok;
        // printf("at %d-th thread\n", omp_get_thread_num());
    while (true) {
    // Block waiting to read the next event from the completion queue. The
    // event is uniquely identified by its tag, which in this case is the
    // memory address of a CallData instance.
    // The return value of Next should always be checked. This return value
    // tells us whether there is any kind of event or cq_ is shutting down.
    CHECK(cq_->Next(&tag, &ok));
    //GPR_ASSERT(ok);
    if (!ok)  continue;
    static_cast<Call*>(tag)->Proceed();
    }
}


// ABSL_FLAG(std::string, data, "/wqwei/deep/base.100M.fbin", "");
// ABSL_FLAG(std::string, query, "/wqwei/deep/query.public.10K.fbin", "");
// ABSL_FLAG(std::string, gt, "/wqwei/deep/deep100M_groundtruth.ivecs", "");

// ABSL_FLAG(std::string, data, "../../deep1b/base.1B.fbin", "");
// ABSL_FLAG(std::string, query, "../../deep1b/query.public.10K.fbin", "");
// ABSL_FLAG(std::string, gt, "../../deep1b/groundtruth.public.10K.ibin", "");
// ABSL_FLAG(int32_t, bin_format, 1, "");

ABSL_FLAG(std::string, load, "", "");
ABSL_FLAG(int32_t, check_k, 10, "# of checked nnns per query");
ABSL_FLAG(double, recall, 0.9, "required recall");
ABSL_FLAG(int32_t, nthreads, 1, "# of threads used in each server");

int main(int argc, char** argv){

    absl::ParseCommandLine(argc, argv);
    if(strcmp(absl::GetFlag(FLAGS_load).c_str(), "") == 0){
        printf("Require a index path. Use --load <path>\n");
        return 0;
    }

    std::string prefix = absl::GetFlag(FLAGS_load) + "/";
    
    const std::string filePath = prefix + "conf.json";


    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if(rank != 0){
        auto cfg = Config(filePath);
        hnswlib::HierarchicalNSW<float>* alg_hnsw;
        int d = cfg.get<int>("dim");
        hnswlib::L2Space space(d);
        int nlist = cfg.get<int>("nlist");
        int gid = (rank-1)%nlist;
        std::string hnsw_path = prefix + "hnsw_"+std::to_string(gid)+".bin";
        std::cout<<rank<<": read from file "<<hnsw_path<<std::endl;
        alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);
        std::cout<<rank<<": read from file OK "<<hnsw_path<<std::endl;
        
        int nns = absl::GetFlag(FLAGS_check_k); // # of nearest neighbor each server
        int &n_res_per_server = nns;
        printf("rank %d graph %d use %d thread\n", rank, gid, omp_get_max_threads()/(world_size-1));
        MPI_Barrier(MPI_COMM_WORLD); // MPI_Barrier2 : prepare OK, start to search
        #pragma omp parallel // num_threads(omp_get_max_threads()/(world_size-1))
        {
            while (true){
                float *vec = new float[d+1];
                MPI_Request req_v;
                MPI_Irecv(vec, d+1, MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &req_v);
                
                hnsw_mpi_server<false, true>(&req_v, n_res_per_server, alg_hnsw, vec, rank);
            }
        }
    }else{
        
        auto cfg = Config(filePath);
        int nlist = cfg.get<int>("nlist");
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
        
        MPI_Barrier(MPI_COMM_WORLD); // MPI_Barrier2

        std::string filename = prefix + "ef_file.csv";
        FILE* fp = fopen(filename.c_str(), "r");
        
        int ief[nlist] = {0};
        int ief_sel[nlist] = {0};
        int nsearch = 0; // init
        int nsearch_sel = -1;
        double tgt = GetFlag(FLAGS_recall);
        printf("Requrie recall larger than %lf\n", tgt);
        double selected_recall = -1.0;
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
            }
            if(read == EOF){
                break;
            }
            fscanf(fp, "%lf", &exp_tgt);
            if(exp_tgt > tgt){
                selected_recall = exp_tgt;
                memcpy(ief_sel, ief, sizeof(int) * nlist);
                nsearch_sel = nsearch;
                break;
            }
        }
        printf("Selected expected recall is %lf,\n ef pre server: ", selected_recall);
        for(int i=0;i<nsearch_sel;++i){
            printf("%d ", ief_sel[i]);
        }
        puts("");
        auto algo = load_from_file(prefix);
        algo->nsearch = nsearch_sel;
        algo->ief = new int[nlist];
        memcpy(algo->ief, ief_sel, sizeof(int)*nlist);
        GrpcAsyncServer server(std::string(""), algo);
        server.Run(50001, GetFlag(FLAGS_nthreads));
    }

    MPI_Abort(MPI_COMM_WORLD, 0);
    MPI_Finalize();
    return 0;
    //float* Dr = new float[nns * nq];
    
}
