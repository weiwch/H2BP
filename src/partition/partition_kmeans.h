#pragma once
#include "partition.h"
#include <stddef.h>
#include <stdint.h>
#include <vector>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/Index.h>

class KmeansPartition: public PartitionAlgo
{
public:
    KmeansPartition(size_t d_, size_t nlist_, float* centroids_ = nullptr);
    ~KmeansPartition();
    static std::shared_ptr<KmeansPartition> load(std::string sav);
    void train(size_t n, const  float *data) override;
    void partition(size_t n, const  float *data, idx_t *res, std::string sav) override;
    //override 
    void rank(size_t n, const float *query, size_t k, idx_t *res) override;
    void save(std::string sav) override;
    faiss::IndexFlat quantizer;
};
