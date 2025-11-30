#include "utils.h"

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include <vector>
#include "absl/log/check.h"
#include "datasets.h"
#include "client.h"
#include "client_async.h"

double compute_recall(const Dataset& ds, const int* Ir, int nns) {
    int total_hits = 0;
    for (int i = 0; i < ds.nq; ++i) {
        std::set<idx_t> gt_set;
        for (int j = 0; j < nns; ++j) {
            gt_set.insert(ds.gt[i * ds.gt_k + j]);
        }
        int hit = 0;
        for (int j = 0; j < nns; ++j) {
            if (gt_set.count(Ir[i * nns + j])) {
                hit++;
            }
        }
        total_hits += hit;
    }
    double recall = 1.0 * total_hits / (ds.nq * nns);
    printf("recall@%d : %d %.5f\n", nns, total_hits, recall);
    return recall;
}

ABSL_FLAG(std::string, query, "../../sift1M/sift_query.fvecs", "");
ABSL_FLAG(std::string, gt, "../../sift1M/sift_groundtruth.ivecs", "");

// ABSL_FLAG(std::string, query, "../../deep/query.public.10K.fbin", "");
// ABSL_FLAG(std::string, gt, "../../deep/deep100M_groundtruth.ivecs", "");

ABSL_FLAG(int32_t, check_k, 10, "# of checked nnns per query");
ABSL_FLAG(bool, async, true, "use async client");
ABSL_FLAG(int32_t, batchsize, 0, "Number of queries per batch RPC");

ABSL_FLAG(bool, insert_test, false, "");

ABSL_FLAG(std::string, data, "../../sift1M/sift_base.fvecs", "");
ABSL_FLAG(std::string, dyn_gt_prefix, "sift_base_", "prefix of dynamic groundtruth file");
ABSL_FLAG(int32_t, dyn_gt_start, 800, "start id of dynamic groundtruth");
ABSL_FLAG(int32_t, dyn_gt_interval, 50, "interval of dynamic groundtruth");
ABSL_FLAG(int32_t, dyn_gt_count, 4, "number of dynamic groundtruth files");

int main(int argc, char** argv){

    absl::ParseCommandLine(argc, argv);

    Dataset ds("", absl::GetFlag(FLAGS_query), absl::GetFlag(FLAGS_gt));
    ds.read_query();

    auto channel_str = "localhost:"+std::to_string(50001);
    
    int nns = absl::GetFlag(FLAGS_check_k);
    int* Ir = new int[nns * ds.nq];

    bool inserttest = absl::GetFlag(FLAGS_insert_test);
    if(inserttest){

        int cnt = absl::GetFlag(FLAGS_dyn_gt_count);
        int start = absl::GetFlag(FLAGS_dyn_gt_start);
        int interval = absl::GetFlag(FLAGS_dyn_gt_interval);
        
        AnnsClient cli = AnnsClient(grpc::CreateChannel(channel_str, grpc::InsecureChannelCredentials()));
        for(int i=0;i<cnt;++i){
            
            std::string gt_file = absl::GetFlag(FLAGS_dyn_gt_prefix) + std::to_string(start + i * interval) + "k.ibin";
            // cli.Insert(gt_file, 1000, ds.d, 1000000, &r);
            size_t dout,nout;
            int *ins_vec = ibin_read(gt_file.c_str(), &dout, &nout);

            int start_id = (start + i*interval)*1000;
            float* queries = fbin_read(absl::GetFlag(FLAGS_data).c_str(), &ds.d, &ds.nb, start_id, interval);
            ResultOfInsert r;
            cli.Insert(queries, nout, ds.d, start_id, &r);
            std::cout<<r.status()<<std::endl;

            delete[] queries;
            delete[] ins_vec;
        }
        int n_insert = 1000;
        
        // cli.Insert(ds.xq, n_insert, ds.d, 1000000, &r);
        std::cout<<"Inserted "<<n_insert<<" vectors."<<std::endl;
        // delete[] ins_vec;
        
        return 0;
    }


    auto start_time = time_now();
    int batch_size = absl::GetFlag(FLAGS_batchsize);
    if(absl::GetFlag(FLAGS_async)==false){
        AnnsClient cli = AnnsClient(grpc::CreateChannel(channel_str, grpc::InsecureChannelCredentials()));
        if(batch_size == 0){
            #pragma omp parallel for
            for(int i=0;i<ds.nq;++i){
                ResultOfQuery r;
                cli.Query(ds.xq + i*ds.d, ds.d, nns, &r);
                // for(int i=0;i<10;++i){
                //     printf("%d\n",r.ids(i));
                // }
                memcpy(Ir+nns*i,r.ids().data(),nns*sizeof(float));
                //printf("%d\n", i);
            }
        }
        else{
            int n_batches = (ds.nq + batch_size - 1) / batch_size;
            // n_batches =2;
            for(int b=0;b<n_batches;++b){
                int current_batch_size = std::min<int>(batch_size, ds.nq - b * batch_size);
                BatchResult r;
                cli.BatchQuery(ds.xq + b * batch_size * ds.d, current_batch_size, ds.d, nns, &r);
                // Process each query result in the batch
                for(int i=0; i<current_batch_size; ++i){
                    const auto& single_result = r.results(i);
                    memcpy(Ir + nns * (b * batch_size + i), single_result.ids().data(), nns * sizeof(uint32_t));
                }
            }
        }
        std::cout<<elapsed_ms(start_time)/ds.nq<<" ms pre query"<<std::endl;
    }
    else{
        

        
        if(batch_size == 0){
            AnnsClientAsync cli(grpc::CreateChannel(channel_str, grpc::InsecureChannelCredentials()), ds.nq);
            std::thread thread_ = std::thread(&AnnsClientAsync::AsyncCompleteRpc, &cli, nns, ds.nq, Ir);
            for(int i=0;i<ds.nq;++i){
                cli.Query(ds.xq + i*ds.d, ds.d, nns, i);
            }
            std::cout<<elapsed_ms(start_time)/ds.nq<<" ms (sent) pre query"<<std::endl;
            thread_.join();
        }
        else{
            int n_batches = (ds.nq + batch_size - 1) / batch_size;
            // n_batches =2;
            AnnsClientAsync cli(grpc::CreateChannel(channel_str, grpc::InsecureChannelCredentials()), n_batches);
            std::thread batch_thread = std::thread(&AnnsClientAsync::AsyncCompleteBatchRpc, &cli, nns, n_batches, Ir);
            for(int b=0;b<n_batches;++b){
                int current_batch_size = std::min<int>(batch_size, ds.nq - b * batch_size);
                cli.BatchQuery(ds.xq + b * batch_size * ds.d, current_batch_size, ds.d, nns, b);
                // break;
            }
            std::cout<<elapsed_ms(start_time)/ds.nq<<" ms (sent) pre query"<<std::endl;
            batch_thread.join();
        }

        
    }
    std::cout<<elapsed_ms(start_time)/ds.nq<<" ms pre query"<<std::endl;

    compute_recall(ds, Ir, nns);
    // fprintf(fp_res, "%lf %lf\n", 1.0*shit/(nq*nns), search_time/nq);
    return 0;
}