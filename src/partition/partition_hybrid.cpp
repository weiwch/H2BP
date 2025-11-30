#include "partition_hybrid.h"
#include <queue>
#include <utility>
#include <set>
#include "config.h"
#include <faiss/index_io.h>
typedef std::pair<float, std::pair<int, int> > tuple3;
typedef std::pair<int, int> pii;

void balanced_assign(faiss::IndexFlatL2* quantizer, size_t nb, size_t d, float* xb, int nlist, size_t n, idx_t* balanced_assg, float b_factor = 1.05, int* count = nullptr, bool test = true, bool weighted = true){

    static std::priority_queue<tuple3, std::vector<tuple3>, std::greater<tuple3> > pq;
    while (!pq.empty())
    {
        pq.pop();
    }
    idx_t* assg = new idx_t[nb * nlist];
    float *dist = new float[nb * nlist];
    quantizer->search(nb, xb, nlist, dist, assg);
    printf("search OK, mode %s\n", weighted ? "weighted" : "unweighted");
    int* cnt = new int[nb]{};
    int* n_each_node = new int[nlist]{};
    double lim;
    if(weighted && count){
        lim = b_factor / (double)nlist  * (double)n;
    }else{
        lim = (b_factor * nb) / nlist;
    }
    printf("lim: %lf\n", lim);
    size_t assg_count = 0;
    if (weighted && count){
        for(int i=0; i<nb; ++i){
            // printf("%f\n", dist[i*nlist]);
            pq.push(tuple3(dist[i*nlist], pii(i, assg[i*nlist])));
        }
        while (assg_count < n)
        {
            auto t = pq.top().second;
            pq.pop();
            int idx = t.first;
            int c = count[idx];
            if(n_each_node[t.second] <= lim){
                balanced_assg[idx] = t.second;
                n_each_node[t.second] += c;
                assg_count += c;
            }
            else{
                int c = ++cnt[idx];
                if(c==nlist){
                    balanced_assg[idx] = t.second;
                    n_each_node[t.second] += c;
                    assg_count += c;
                }else{
                    pq.push(tuple3(dist[idx*nlist+c], pii(idx, assg[idx*nlist+c])));
                }
            }
        }
    }else{
        for(int i=0; i<nb; ++i){
        // printf("%f\n", dist[i*nlist]);
            pq.push(tuple3(dist[i*nlist], pii(i, assg[i*nlist])));
        }
        while (assg_count < nb)
        {
            auto t = pq.top().second;
            pq.pop();
            int idx = t.first;
            if(n_each_node[t.second] < lim){
                balanced_assg[idx] = t.second;
                ++n_each_node[t.second];
                ++assg_count;
            }
            else{
                int c = ++cnt[idx];
                pq.push(tuple3(dist[idx*nlist+c], pii(idx, assg[idx*nlist+c])));
            }
        }
    }
    if(test){
        int sum = 0;
        for(int i=0;i<nlist;++i){
            printf("%d ",n_each_node[i]);
            sum+=n_each_node[i];
        }
    
        puts("");
        printf("lim: %lf, sum %d\n", lim, sum);
        // calc stdev:
        double mean = sum / nlist;
        double sq_sum = 0.0;
        for(int i=0;i<nlist;++i){
            sq_sum += (n_each_node[i] - mean) * (n_each_node[i] - mean);
        }
        double stdev = sqrt(sq_sum / nlist);
        printf("stddev: %lf, mean: %lf, ratio(cv): %lf\n", stdev, mean, stdev / mean);
    }
    delete[] n_each_node;
    delete[] dist;
    delete[] assg;
    delete[] cnt;
    quantizer->reset();
    float* hassg = new float[nlist];
    float* cent = new float[nlist * d];
    faiss::compute_centroids(d, nlist, nb, 0, (uint8_t*)xb, nullptr, balanced_assg, nullptr, hassg, cent);
    quantizer->add(nlist, cent);
    delete[] cent;
    delete[] hassg;
    // for(int i=0;i<nlist;++i){
    //     printf("%d:", i);
    //     for(int j=0;j<d;++j){
    //         printf("%.3f ", cent[i*d +j]);
    //     }
    //     puts("");
    // }
}

HybridTreePartition::HybridTreePartition(size_t d_, size_t nlist_, size_t nsublist_, float* centroids_, bool weighted_, bool balanced_) : PartitionAlgo(d_, nlist_), nsublist(nsublist_), quantizer(d_), top_quantizer(d_), map_to_top(nsublist_, 0)
{
    if (centroids_) {
        quantizer = faiss::IndexFlatL2(d);
        quantizer.add(nlist, centroids_);
    } else {
        quantizer = faiss::IndexFlatL2(d);
    }
    quantizer.verbose = false;
    top_quantizer.verbose = false;
    weighted = weighted_;
    balanced = balanced_;
    count = new int[nsublist]{}; // for test only, init count forever
    // if(weighted_){
    //     count = new int[nsublist]{};
    // }
    // else{
    //     count = nullptr;
    // }
}

HybridTreePartition::~HybridTreePartition()
{

}

std::shared_ptr<HybridTreePartition> HybridTreePartition::load(std::string sav){
    const std::string filePath = sav + "conf.json";
    auto cfg = Config(filePath);
    auto p = std::make_shared<HybridTreePartition>(cfg.get<int>("dim"), cfg.get<int>("nlist"), cfg.get<int>("nsublist"));
    p->quantizer = *static_cast<faiss::IndexFlatL2*>(faiss::read_index((sav + "quantizer.index").c_str()));
    p->top_quantizer = *static_cast<faiss::IndexFlatL2*>(faiss::read_index((sav + "top_quantizer.index").c_str()));
    p->routing_method = cfg.get<int>("routing_method");
    p->partition_method = cfg.get<int>("partition_method");
    p->map_to_top.resize(p->nsublist);
    std::string map_file = sav + "map_to_top.bin";
    FILE *fp = fopen(map_file.c_str(), "rb");
    if (fp == nullptr) {
        throw std::runtime_error("Failed to open map_to_top file: " + map_file);
    }
    fread(p->map_to_top.data(), sizeof(idx_t), p->nsublist, fp);
    //print map_to_top
    printf("loaded map_to_top: ");
    for(auto x : p->map_to_top){
        printf("%d ", x);
    }
    printf("\n");
    fclose(fp);
    printf("HybridTreePartition loaded from %s\n", sav.c_str());
    return p;
}

void HybridTreePartition::train(size_t n, const float *data)
{
    // get centroid of the clusters
    float *centroid = new float[nsublist * d];
    if(quantizer.ntotal == nlist){
        printf("HBTreePartition already trained.\n");
    }
    else{
        auto index = faiss::IndexIVFFlat(&quantizer, d, nsublist);
        index.train(n, data);
    }
    if(count){
        
        idx_t *assi = new idx_t[n];
        quantizer.assign(n, data, assi);
        // TODO: consider to reuse count function of PartitionAlgo
        for(int i=0;i<n;i++){
            count[assi[i]]++;
        }
        delete[] assi;
        printf("sublist count: ");
        for(int i=0;i<nsublist;i++){
            printf("%d ", count[i]);
        }
        printf("\n");
    }
    quantizer.reconstruct_n(0, nsublist, centroid);
    // normal kmeans
    
    if(balanced){
        auto index2 = faiss::IndexIVFFlat(&top_quantizer, d, nlist);
        index2.train(nsublist, centroid);
        // int it = 3;
        // while(it--)
        balanced_assign(&top_quantizer, nsublist, d, centroid, nlist, n, map_to_top.data(), 1.005, count, true, weighted);
    }
    else{
        printf("balanced is false, not balanced assign\n");
        if(weighted && count){
            printf("weighted balanced assign\n");
            auto cluster = faiss::Clustering(d, nlist);
            cluster.verbose = false;
            cluster.niter = 20;
            float *tmp = new float[nsublist];
            for(int i=0;i<nsublist;++i){
                tmp[i] = (float)count[i];
            }
            cluster.train(nsublist, centroid, top_quantizer, tmp);
            delete[] tmp;
            top_quantizer.assign(nsublist, centroid, map_to_top.data());
        }else{
            auto index2 = faiss::IndexIVFFlat(&top_quantizer, d, nlist);
            index2.train(nsublist, centroid);

            // float *top_centroid = new float[nlist * d];
            // top_quantizer.reconstruct_n(0, nlist, top_centroid);
            top_quantizer.assign(nsublist, centroid, map_to_top.data());
        }
    }
    // random assign
    // for(int i=0;i<nsublist;i++){
    //     map_to_top[i] = i % nlist;
    // }
    
    for(auto x : map_to_top){
        printf("%d ", x);
    }
    printf("\n");
    delete[] centroid;
}

void HybridTreePartition::partition(size_t n, const float *data, idx_t *res, std::string sav)
{
    if(partition_method == 0){
        top_quantizer.assign(n, data, res);
    }
    else{
        quantizer.assign(n, data, res);
        // printf("before map_to_top: ");
        // for(int i=0;i<1000;i++){
        //     printf("%d ", res[i]);
        // }
        #pragma omp parallel for 
        for(int i=0;i<n;i++){
            res[i] = map_to_top[res[i]];
        }
    }
    if (!sav.empty()) {
        // save_partition(sav, n, res);
        save(sav);
    }
}

void HybridTreePartition::rank(size_t n, const float *query, size_t k, idx_t *res)
{
    if(routing_method == 0 ){
        top_quantizer.assign(n, query, res, k);
    }
    else{
        int kk = nsublist / nlist;// k;// nsublist / nlist * k;
        idx_t *tmp_res = new idx_t[kk*n];
        printf("kk: %d\n", kk);
        fflush(stdout);
        quantizer.assign(n, query, tmp_res, kk);
        #pragma omp parallel for 
        for(int i=0;i<n;i++){
            std::set<int> st;
            int p = 0;
            for(int j=0;j<kk;j++){
                auto now_top_parti = map_to_top[tmp_res[i*kk+j]];
                if(st.count(now_top_parti) == 0){
                    st.insert(now_top_parti);
                    res[i*k+(p++)] = now_top_parti;
                    if(p==k){
                        break;
                    }
                }
            }
            for(int j=p;j<k;j++){
                res[i*k+j] = -1;
            }
        }
        delete[] tmp_res;
    }
}

void HybridTreePartition::save(std::string sav)
{
    // save map_to_top
    std::string map_file = sav + "map_to_top.bin";
    FILE *fp = fopen(map_file.c_str(), "wb");
    if (fp == nullptr) {
        throw std::runtime_error("Failed to open map_to_top file: " + map_file);
    }
    fwrite(map_to_top.data(), sizeof(idx_t), nsublist, fp);
    fclose(fp);
    faiss::write_index(&quantizer, (sav + "quantizer.index").c_str());
    faiss::write_index(&top_quantizer, (sav + "top_quantizer.index").c_str());
    printf("Quantizer and top quantizer saved to %s and %s\n", 
            (sav + "quantizer.index").c_str(),
            (sav + "top_quantizer.index").c_str());  
    ConfigWriter writer;
    writer.set("dim", d);
    writer.set("nlist", nlist);
    writer.set("nsublist", nsublist);
    writer.set("routing_method", routing_method);
    writer.set("partition_method", partition_method);
    writer.set("method", "hb");
    writer.save(sav + "conf.json");
    printf("Hybrid partition saved to %s\n", (sav + "conf.json").c_str());
}
