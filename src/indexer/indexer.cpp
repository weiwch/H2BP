#include "datasets.h"
#include "partition/partition.h"
#include "partition/partition_kmeans.h"
#include "partition/partition_hybrid.h"
#include "partition/partition_graph.h"
#include "partition/partition_graphp.h"
#include <iostream>
#include <string>
#include <omp.h>
#include <vector>
#include <set>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include <filesystem>
namespace fs = std::filesystem;

ABSL_FLAG(std::string, partition, "kmeans", "rand, kmeans, hb, graph, g2");
ABSL_FLAG(std::string, data, "../../sift1M/sift_base.fvecs", "");
ABSL_FLAG(std::string, query, "../../sift1M/sift_query.fvecs", "");
ABSL_FLAG(std::string, gt, "../../sift1M/sift_groundtruth.ivecs", "");
// ABSL_FLAG(std::string, data, "../../webvid_split/clip.webvid.base.2.5M.fbin", "");
// ABSL_FLAG(std::string, query, "../../webvid_split/sift_query.fvecs", "");
// ABSL_FLAG(std::string, gt, "../../webvid_split/sift_groundtruth.ivecs", "");

ABSL_FLAG(int32_t, nlist, 3, "");
ABSL_FLAG(int32_t, nsublist, 9, "");
ABSL_FLAG(int32_t, knn, 10, "");
ABSL_FLAG(float_t, factor, 1.00001, "");
ABSL_FLAG(bool, weighted, true, "");
ABSL_FLAG(bool, balanced, true, "");
ABSL_FLAG(std::string, save, "idx_sav", "");
ABSL_FLAG(bool, export, true, "");

ABSL_FLAG(int32_t, rec_st, 1, "");
ABSL_FLAG(int32_t, rec_ed, -1, "");

int main(int argc, char** argv){
    absl::ParseCommandLine(argc, argv);

    std::string prefix = absl::GetFlag(FLAGS_save) + "/";

    try {
        if (fs::exists(prefix)) {
            throw std::runtime_error("Folder already exists!");
        }
        if (fs::create_directory(prefix)) {
            std::cout << "Folder created successfully: " << prefix << std::endl;
        } else {
            std::cerr << "Failed to create folder: " << prefix << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    int nlist = absl::GetFlag(FLAGS_nlist);
    int knn = absl::GetFlag(FLAGS_knn);
    Dataset ds(absl::GetFlag(FLAGS_data), absl::GetFlag(FLAGS_query), absl::GetFlag(FLAGS_gt));
    ds.read_data();
    // shared ptr for PartitionAlgo
    std::shared_ptr<PartitionAlgo> partition;
    std::string partition_method = absl::GetFlag(FLAGS_partition);
    // if exist perfix + "/centroids.fbin"

    float* centroids = nullptr;
    
    size_t now_d, now_n;
    if(fs::exists(prefix + "centroids.fbin")){
        printf("LOAD centroids from %s\n", (prefix + "centroids.fbin\n").c_str());
        centroids = fbin_read((prefix + "centroids.fbin").c_str(), &now_d, &now_n);
        if(now_d != ds.d){
            throw std::invalid_argument("centroids dimension or number not match");
        }
    }
    
    if(partition_method == "rand"){
        partition = std::make_shared<RandPartition>(ds.d, nlist, 0);
    }else if(partition_method == "kmeans"){
        if(centroids && now_n != nlist){
            throw std::invalid_argument("centroids number not match nlist");
        }
        partition = std::make_shared<KmeansPartition>(ds.d, nlist, centroids);
    }else if(partition_method == "hb"){

        int nsublist = absl::GetFlag(FLAGS_nsublist);
        if(centroids && now_n != nsublist){
            throw std::invalid_argument("centroids number not match nsublist");
        }
        auto p = std::make_shared<HybridTreePartition>(ds.d, nlist, nsublist);
        p->routing_method = 1;
        p->partition_method = 1;
        p->balanced = absl::GetFlag(FLAGS_balanced);
        p->weighted = absl::GetFlag(FLAGS_weighted);
        partition = p;
    }else if(partition_method == "hb_t"){
        int nsublist = absl::GetFlag(FLAGS_nsublist);
        if(centroids && now_n != nsublist){
            throw std::invalid_argument("centroids number not match nsublist");
        }
        auto p = std::make_shared<HybridTreePartition>(ds.d, nlist, nsublist);
        p->routing_method = 0;
        p->partition_method = 0;
        p->balanced = absl::GetFlag(FLAGS_balanced);
        p->weighted = absl::GetFlag(FLAGS_weighted);
        partition = p;
    }else if(partition_method == "graph"){
        auto p = std::make_shared<GraphPartition>(ds.d, nlist);
        // p->ef = absl::GetFlag(FLAGS_nsublist) * 5; // set ef for search (meta hnsw)
        p->nsublist = absl::GetFlag(FLAGS_nsublist); // 100 is good in paper // we borrow the value to use as branching factor K (the number of top neighbors in the meta-HNSW that are used to choose the sub-HNSWs for a query. 
        partition = p;
    }else if(partition_method == "gp"){
        int nsublist = absl::GetFlag(FLAGS_nsublist);
        auto p = std::make_shared<GraphPartition2>(ds.d, nlist, nsublist);
        // copy string data path
        p->datapath = absl::GetFlag(FLAGS_data);
        partition = p;
    }else{
        //throw error
        throw std::invalid_argument("Unknown partition method: " + partition_method);
    }

    auto start = time_now();
    partition->train(ds.nb, ds.xb);
    printf("train OK, cost %lf ms\n", elapsed_ms(start));
    // return 0;
    idx_t* label = new idx_t[ds.nb]; 
    start = time_now();
    partition->partition(ds.nb, ds.xb, label, prefix);
    printf("partition OK, cost %lf ms\n", elapsed_ms(start));
    // prefix + "partition.idx"
    partition->save_partition(prefix + "partition.idx", ds.nb, label);
    auto cnt = partition->count(ds.nb, label);
    partition->main_stdev(cnt, true);
    
    int ed = absl::GetFlag(FLAGS_rec_ed);
    int st = absl::GetFlag(FLAGS_rec_st);
    //
// MPI_Abort(MPI_COMM_WORLD, 0); // 
    //
    int n_test_list = ed - st + 1;
    if(n_test_list > 0){
        
        ds.read_query();
        std::vector<std::set<idx_t> > ivf_set(nlist);
        for(int i=0;i<ds.nb;++i){
            ivf_set[label[i]].insert(i);
        }

        idx_t* res = new idx_t[ds.nq * ed];
        start = time_now();
        partition->rank(ds.nq, ds.xq, ed, res);
        printf("rank OK, cost %lf ms\n", elapsed_ms(start));
        
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
        // vector of size [n_test_list, nlist]
                        // hit_count[i-st][res[j*ed+l]] += found;
        std::vector<std::vector<size_t> > touch_count(n_test_list, std::vector<size_t>(nlist, 0));
        std::vector<std::vector<size_t> > hit_count(n_test_list, std::vector<size_t>(nlist, 0));
        for(int j=0;j<ds.nq;++j){
            for(int i = st; i <= ed; i++){
                int l = i-1;
                if(res[j*ed+l] == -1){
                    break;
                }
                auto &now_set = ivf_set[res[j*ed+l]];
                touch_count[i-st][res[j*ed+l]] += 1;

                for(int k = 0; k < knn; ++k){
                    int found = 0;
                    idx_t id = ds.gt[j*ds.gt_k+k];
                    if(now_set.count(id)){
                        hit_count[i-st][res[j*ed+l]] += 1;
                    }
                }
            }
        
        }
        for(int i=0;i<n_test_list;++i){
            printf("Detail for %d nodes:\n", i + st);
            for(int j=0;j<nlist;++j){
                if(touch_count[i][j] > 0){
                    double recall_val = (double)hit_count[i][j] / (knn * touch_count[i][j]);
                    double avg_touch = (double)touch_count[i][j] / ds.nq;
                    printf("  list %d: touch %zu, hit %zu, recall %lf, avg_touch %lf\n", j, touch_count[i][j], hit_count[i][j], recall_val, avg_touch);
                }
            }
        }
        // print recall avg touch in csv format
        for(int i=0;i<n_test_list;++i){
            for(int j=0;j<nlist;++j){
                if(touch_count[i][j] > 0){
                    double recall_val = (double)hit_count[i][j] / (knn * touch_count[i][j]);
                    printf("%lf,", recall_val);
                }
            }
            printf("\n");
            for(int j=0;j<nlist;++j){
                if(touch_count[i][j] > 0){
                    double avg_touch = (double)touch_count[i][j] / ds.nq;
                    printf("%lf,", j, avg_touch);
                }
            }
            printf("\n");
        }

        for(int i=0;i<n_test_list;++i){
            double recall_val = (double)sum_hit[i] / (knn * ds.nq);
            double avg_touch = (double)sum_touch[i] / ds.nq;
            // Scaled Recall per Calculated Vector 
            // Recall per Processed Database Fraction
            // normalized recall per vector
            printf("recall of %d nodes, calc # of vevc %lf: %lf (srpv %lf)\n", i + st, avg_touch, recall_val, recall_val / avg_touch * ds.nb);
        }
    }
    if(absl::GetFlag(FLAGS_export)){
        // export data to prefix + "pack_" + list_id + ".fvecs"
        // and prefix + "id_" + list_id + ".bin"
        printf("export data to %spack_*.fvecs and %sid_*.bin\n", prefix.c_str(), prefix.c_str());
        FILE* files[nlist];
        FILE* id_files[nlist];
        std::vector<int> ivmap[nlist]; // not used

        for (int i = 0; i < nlist; i++) {
            char filename[100];
            sprintf(filename, "%spack_%d.fvecs", prefix.c_str(), i);
            files[i] = fopen(filename, "wb");
            if (!files[i]) {
                printf("can not open %s\n", filename);
                return -1;
            }
            sprintf(filename, "%sid_%d.bin", prefix.c_str(), i);
            id_files[i] = fopen(filename, "wb");
            if (!id_files[i]) {
                printf("can not open %s\n", filename);
                return -1;
            }
        }
        for (int i = 0; i < ds.nb; i++) {
            int list_id = label[i];
            fwrite(&ds.d, sizeof(float), 1, files[list_id]);
            fwrite(&ds.xb[i * ds.d], sizeof(float), ds.d, files[list_id]);
            fwrite(&i, sizeof(int), 1, id_files[list_id]);
            ivmap[list_id].push_back(i);
        }
        // print number of vectors in each partition
        for (int i = 0; i < nlist; i++) {
            fclose(files[i]);
            fclose(id_files[i]);
            printf("partition %d has %zu vectors\n", i, ivmap[i].size()); 
        }
        fflush(stdout);
    }
    delete[] label;
    if(centroids) delete[] centroids;
    return 0;
}
