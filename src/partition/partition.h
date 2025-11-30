#pragma once
#include <stddef.h>
#include <stdint.h>
#include <vector>
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/Index.h"

using idx_t = int64_t;

class PartitionAlgo
{
public:
    PartitionAlgo(size_t d_, size_t nlist_ = 0);
    virtual ~PartitionAlgo();
    virtual void train(size_t n, const float *data);
    virtual void partition(size_t n, const float *data, idx_t *res, std::string sav) = 0;
    virtual void rank(size_t n, const float *query, size_t k, idx_t *res) = 0;
    virtual void save(std::string sav) = 0;
    std::vector<size_t> count(size_t n,const idx_t *res);
    std::pair<double, double> main_stdev(std::vector<size_t>& cnt, bool verbose = false);
    void save_partition(const std::string& path, size_t n, idx_t *parti);
    idx_t* load_partition(const std::string& file_name, size_t* nptr = nullptr);
    size_t d;
    size_t* n_server;
    size_t nsearch;
    int *ief;
    size_t nlist;
};

class RandPartition: public PartitionAlgo
{
public:
    RandPartition(size_t d_, size_t nlist_, unsigned seed = 0);
    ~RandPartition();
    void partition(size_t n, const float *data, idx_t *res, std::string sav) override;
    void rank(size_t n, const float *query, size_t k, idx_t *res) override;
    void save(std::string sav) override;
};
