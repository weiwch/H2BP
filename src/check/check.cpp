#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include "datasets.h"
#include <omp.h>
#include "hnswlib/hnswlib.h"
#include <string>
#include <iostream>
#include "utils.h"
#include <set>

#include "partition/partition.h"
#include "partition/partition_kmeans.h"
#include "partition/partition_hybrid.h"
#include "partition/partition_graph.h"
#include "partition/partition_loader.h"
#include "config.h"
#include <filesystem>
using faiss::idx_t;

ABSL_FLAG(std::string, load, "", "");
// ABSL_FLAG(std::string, data, "../../deep/base.100M.fbin", "");
// ABSL_FLAG(std::string, query, "../../deep/query.public.10K.fbin", "");
// ABSL_FLAG(std::string, gt, "../../deep/deep100M_groundtruth.ivecs", "");

ABSL_FLAG(std::string, data, "../../sift1M/sift_base.fvecs", "");
ABSL_FLAG(std::string, query, "../../sift1M/sift_query.fvecs", "");
ABSL_FLAG(std::string, gt, "../../sift1M/sift_groundtruth.ivecs", "");
ABSL_FLAG(int32_t, knn, 10, "# of checked nnns per query");

ABSL_FLAG(int32_t, smpl, 0, "gen a table use sample size query points");
ABSL_FLAG(int32_t, start, 50, "");
ABSL_FLAG(int32_t, stop, 1000, "");
ABSL_FLAG(int32_t, interval, 50, "");

ABSL_FLAG(int32_t, rec_st, 1, "");
ABSL_FLAG(int32_t, rec_ed, 3, "");

const bool record_time_src = false;

int main(int argc, char** argv){

    absl::ParseCommandLine(argc, argv);

    if(strcmp(absl::GetFlag(FLAGS_load).c_str(), "") == 0){
        printf("Require a index path. Use --load <path>\n");
        return 0;
    }


    int knn = absl::GetFlag(FLAGS_knn);
    Dataset ds(absl::GetFlag(FLAGS_data), absl::GetFlag(FLAGS_query), absl::GetFlag(FLAGS_gt));
    ds.read_data();
    ds.read_query();

    
    std::string prefix = absl::GetFlag(FLAGS_load) + "/";
    std::shared_ptr<PartitionAlgo> partition = load_from_file(prefix);
    int nlist = partition->nlist;
    // if has file prefix + "partition.idx"
    idx_t *label = nullptr;
    if(std::filesystem::exists(prefix + "partition.idx") ){
        label = partition->load_partition(prefix + "partition.idx", &ds.nb);        
    }else{
        label = new idx_t[ds.nb];
        partition->partition(ds.nb, ds.xb, label, "");
    }
    // for(int i=0;i<100;i++){
    //     printf("%lld ", label[i]);
    // }
    auto cnt = partition->count(ds.nb, label);
    partition->main_stdev(cnt, true);

    int ed = absl::GetFlag(FLAGS_rec_ed);
    int st = absl::GetFlag(FLAGS_rec_st);
    //
    // MPI_Abort(MPI_COMM_WORLD, 0); // 
    //
    int n_test_list = ed - st + 1;
    if(n_test_list > 0){
        std::vector<std::set<idx_t> > ivf_set(nlist);
        for(int i=0;i<ds.nb;++i){
            ivf_set[label[i]].insert(i);
        }

        idx_t* res = new idx_t[ds.nq * ed];
        auto start = time_now();
        partition->rank(ds.nq, ds.xq, ed, res);
        printf("rank OK, cost %lf ms\n", elapsed_ms(start));
        // for(int i=0;i<100;++i){
        //     for(int j=0;j<ed;++j){
        //         printf("%lld ", res[i*ed+j]);
        //     }
        //     puts("");
        // }
        std::vector<size_t> sum_hit(n_test_list, 0);
        std::vector<size_t> sum_touch(n_test_list, 0); 
        for(int j=0;j<ds.nq;++j){
            for(int i = st; i <= ed; i++){
                int hit = 0;
                for(int k = 0; k < knn; ++k){
                    int found = 0;
                    idx_t id = ds.gt[j*ds.gt_k+k];
                    for(int l = 0; l < i ; ++l){
                        if(res[j*ed+l] == -1){
                            break;
                        }
                        auto &now_set = ivf_set[res[j*ed+l]];
                        if(now_set.count(id)){
                            found = 1;
                        }
                    }
                    hit += found;
                }
                for(int l = 0; l < i ; ++l){
                    if(res[j*ed+l] == -1){
                        break;
                    }
                    auto &now_set = ivf_set[res[j*ed+l]];
                    sum_touch[i-st] += now_set.size();
                }
                sum_hit[i-st] += hit;
            }
        }
        for(int i=0;i<n_test_list;++i){
            double recall_val = (double)sum_hit[i] / (knn * ds.nq);
            double avg_touch = (double)sum_touch[i] / ds.nq;
            // Scaled Recall per Calculated Vector 
            // Recall per Processed Database Fraction
            // normalized recall per vector
            printf("recall of %d nodes, calc # of vec %lf: %lf (srpv %lf)\n", i + st, avg_touch, recall_val, recall_val / avg_touch * ds.nb);
        }
    }
    return 0;
}