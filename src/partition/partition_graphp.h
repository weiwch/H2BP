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
#include "routes.h"
#include <string>
#include "kmeans_tree_router.h"

class GraphPartition2: public PartitionAlgo
{
public:
    GraphPartition2(size_t d_, size_t nlist_, int budget);
    ~GraphPartition2();
    void train(size_t n, const float *data) override;
    void partition(size_t n, const float *data, idx_t *res, std::string sav) override;
    //override 
    void rank(size_t n, const float *query, size_t k, idx_t *res) override;
    void save(std::string sav) override;
    std::string index_name = "gpann.index"; // default index name
    // std::string routing_index_name = "gpann.index.routing_index_partition"; // default routing index name
    std::vector<int> routing_partition;
    static std::shared_ptr<GraphPartition2> load(std::string sav);

    KMeansTreeRouterOptions routing_index_options;
    int budget;
    KMeansTreeRouter router;
    bool trained;
    std::string datapath;
};
