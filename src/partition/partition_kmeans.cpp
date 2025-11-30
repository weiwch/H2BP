#include "partition_kmeans.h"
#include <faiss/index_io.h>
#include "config.h"

KmeansPartition::KmeansPartition(size_t d_, size_t nlist_, float* centroids_) : PartitionAlgo(d_, nlist_)
{
    if (centroids_) {
        quantizer = faiss::IndexFlatL2(d);
        quantizer.add(nlist, centroids_);
    } else {
        quantizer = faiss::IndexFlatL2(d);
    }
    quantizer.verbose = false;
}

KmeansPartition::~KmeansPartition()
{
}

std::shared_ptr<KmeansPartition> KmeansPartition::load(std::string sav)
{
    const std::string filePath = sav + "conf.json";
    auto cfg = Config(filePath);
    auto p = std::make_shared<KmeansPartition>(cfg.get<int>("dim"), cfg.get<int>("nlist"));
    p->quantizer = *static_cast<faiss::IndexFlat*>(faiss::read_index((sav + "kmeans.index").c_str()));
    return p;
}

void KmeansPartition::train(size_t n, const float *data)
{
    if(quantizer.ntotal == nlist){
        printf("KmeansPartition already trained.\n");
        return;
    }
    auto index = faiss::IndexIVFFlat(&quantizer, d, nlist);
    index.train(n, data);
}

void KmeansPartition::partition(size_t n, const float *data, idx_t *res, std::string sav)
{
    quantizer.assign(n, data, res);
    // save index
    if (!sav.empty()) {
        save(sav);
    }
}

void KmeansPartition::rank(size_t n, const float *query, size_t k, idx_t *res)
{
    quantizer.assign(n, query, res, k);
}

void KmeansPartition::save(std::string sav)
{
    FILE *fp = fopen((sav+"kmeans.index").c_str(), "wb");
    faiss::write_index(&quantizer, fp);
    fclose(fp);
    printf("Kmeans index saved to %s\n", (sav + "kmeans.index").c_str());
    ConfigWriter writer;
    writer.set("dim", d);
    writer.set("nlist", nlist);
    writer.set("method", "kmeans");
    writer.save(sav + "conf.json");
}
