#pragma once
#include "partition.h"
#include <stddef.h>
#include <stdint.h>
#include <vector>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/Index.h>

class HybridTreePartition: public PartitionAlgo
{
public:
    HybridTreePartition(size_t d_, size_t nlist_, size_t nsublist_, float* centroids_ = nullptr, bool weighted = false, bool balanced = false);
    ~HybridTreePartition();
    static std::shared_ptr<HybridTreePartition> load(std::string sav);
    void train(size_t n, const float *data) override;
    void partition(size_t n, const float *data, idx_t *res, std::string sav) override;
    //override 
    void rank(size_t n, const float *query, size_t k, idx_t *res) override;
    void save(std::string sav) override;
    faiss::IndexFlatL2 quantizer, top_quantizer;
    size_t nsublist;
    int routing_method = 1; // 0: by top, 1: by sub
    int partition_method = 1; // 0: by top, 1: by sub
    int *count;
    bool weighted;
    bool balanced;
    std::vector<idx_t> map_to_top;
};