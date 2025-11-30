#pragma once
#include "partition.h"
#include <stddef.h>
#include <stdint.h>
#include <vector>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/Index.h>
#include <vector>
#include "hnswlib/hnswlib.h"

class GraphPartition: public PartitionAlgo
{
public:
    GraphPartition(size_t d_, size_t nlist_);
    ~GraphPartition();
    void train(size_t n, const float *data) override;
    void partition(size_t n, const float *data, idx_t *res, std::string sav) override;
    //override 
    void rank(size_t n, const float *query, size_t k, idx_t *res) override;
    void save(std::string sav) override;
    std::string index_name = "pyramind.index"; // default index name
    std::string routing_index_name = "pyramind.index.routing_index_partition"; // default routing index name
    int ef = 200; // default ef for search, can be changed by user
    size_t nsublist;
    faiss::IndexFlatL2 top_quantizer; // for test
    std::vector<int> routing_partition;
    hnswlib::L2Space space;
    hnswlib::HierarchicalNSW<float> top_hnsw;
    static std::shared_ptr<GraphPartition> load(std::string sav);
};
