#include "partition_graph.h"
#include "partitioning.h"
#include "metis_io.h"
#include <string>
#include "hnswlib/hnswlib.h"
#include <set>
#include "config.h"
#include <faiss/index_io.h>
#include "utils.h"

GraphPartition::GraphPartition(size_t d_, size_t nlist_): PartitionAlgo(d_, nlist_), top_quantizer(d_), space(d), top_hnsw(&space)
{
}

GraphPartition::~GraphPartition()
{
}

void GraphPartition::train(size_t n, const float *data)
{
    // do nothing
}

void GraphPartition::partition(size_t n, const float *data, idx_t *res, std::string sav)
{

    auto points = PointSet();
    points.n = n;
    points.d = d;
    // copy data to points.coordinates
    points.coordinates.resize(n * d);
    std::memcpy(points.coordinates.data(), data, n * d * sizeof(float));
    // 1 partition
    // 2 hnsw
    // 3 hnsw's partition 
    auto start = time_now();
    auto partition = PyramidPartitioning(points, nlist, 0.05, sav + index_name);
    printf("Partitioning time: %lf s\n", elapsed_s(start));
    fflush(stdout);
    for(int i=0;i<partition.size();i++){
        res[i] = partition[i];
    }
    points.Drop();
    save_partition(sav + "partition.idx", n, res);
    save(sav);

    routing_partition = ReadMetisPartition(sav + routing_index_name);
    top_hnsw.loadIndex(sav + index_name, &space, 10000);
    // float* hassg = new float[nlist];
    // float* cent = new float[nlist * d];
    // faiss::compute_centroids(d, nlist, n, 0, (uint8_t*)data, nullptr, res, nullptr, hassg, cent);
    // top_quantizer.add(nlist, cent);


    // auto partition = OurPyramidPartitioning(points, nlist, 0.05, "our_pyramind2.index");
    // auto partition = GraphPartitioning(points, nlist, 0.05, false);
    
}

void GraphPartition::rank(size_t n, const float *query, size_t k, idx_t *res)
{
    //
    // top_quantizer.assign(n, query, res, k);
    // return;
    
    // Clusters clusters = ReadClusters(partition_file); //nlist == clusters.size()
    // load hnsw index
    
    top_hnsw.setEf(ef); // set ef for search
    printf("top hnsw ef : %d\n", ef);
    // FILE* fp = fopen(routing_index_name, "r");
    // if(fp == nullptr){
    //     throw std::runtime_error("Failed to open routing index file: " + routing_index_name);
    // }
    // printf("%d nsublist(!!)\n", nsublist);

    #pragma omp parallel for
    for(int i=0;i<n;++i){
        std::set<int> st;
        auto res_heap = top_hnsw.searchKnn(query+i*d, nsublist);
        int t=0;
        while (!res_heap.empty())
        {
            auto label = res_heap.top().second;
            res_heap.pop();
            // if (label < 0 || label >= routing_partition.size()) {
            //     continue; // Skip invalid labels
            // }
            int partition_id = routing_partition[label];
            if(st.count(partition_id) == 0){
            // if (partition_id < 0 || partition_id >= nlist) {
            //     continue; // Skip invalid partition IDs
            // }
                st.insert(partition_id);
                res[i * k + (t++)] = partition_id; // Store the label in the result
            }
        }
        std::reverse(res + i * k, res + i * k + t); // Reverse to have closer first
        for(;t<k;++t){
            res[i * k + t] = -1;
        }
    }
    
    
}

void GraphPartition::save(std::string sav)
{
    // save index
    // faiss::write_index(&top_quantizer, (sav + index_name).c_str());
    ConfigWriter writer;
    writer.set("dim", d);
    writer.set("nlist", nlist);
    writer.set("method", "graph");
    writer.set("n", 10000); // static default setting in paper
    writer.set("ef", 200); // ef search
    writer.set("nsublist", 1000); // branch at search, good in paper
    writer.save(sav + "conf.json");
    
}

std::shared_ptr<GraphPartition> GraphPartition::load(std::string sav){
    const std::string filePath = sav + "conf.json";
    auto cfg = Config(filePath);
    auto p = std::make_shared<GraphPartition>(cfg.get<int>("dim"), cfg.get<int>("nlist"));
    p->top_hnsw.loadIndex(sav + p->index_name, &p->space, cfg.get<int>("n"));
    p->ef = cfg.get<int>("ef"); // ef search
    p->routing_partition = ReadMetisPartition(sav + p->routing_index_name);
    p->top_hnsw.ef_ = p->ef; // set ef for search
    p->nsublist = cfg.get<int>("nsublist");
    return p;
}