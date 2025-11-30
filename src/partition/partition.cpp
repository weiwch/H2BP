#include "partition.h"
#include <stdlib.h>
#include <numeric>
#include <algorithm>
#include <omp.h>
#include "config.h"

PartitionAlgo::PartitionAlgo(size_t d_, size_t nlist_): d(d_), nlist(nlist_)
{
    n_server = new size_t[nlist];
    for(int i=0;i<nlist;i++){
        n_server[i] = 1;
    }
}

PartitionAlgo::~PartitionAlgo()
{
    delete[] n_server;
}
void PartitionAlgo::train(size_t n, const float *data)
{
}

std::vector<size_t> PartitionAlgo::count(size_t n, const idx_t *res)
{
    std::vector<size_t> count(nlist, 0);
    for (size_t i = 0; i < n; i++) {
        count[res[i]]++;
    }
    return count;
}

std::pair<double, double> PartitionAlgo::main_stdev(std::vector<size_t> &cnt, bool verbose)
{   
    if(verbose){
        printf("count: ");
        for(auto x : cnt){
            printf("%zu ", x);
        }
        printf("\n");
    }
    double mean = 0;
    for(auto x : cnt){
        mean += x;
    }
    mean /= cnt.size();
    double stddev = 0;
    for(auto x : cnt){
        stddev += (x - mean) * (x - mean);
    }
    stddev = sqrt(stddev / cnt.size());
    if(verbose){
        printf("mean: %lf, stddev: %lf, cv: %lf\n", mean, stddev, stddev / mean);
    }
    return std::pair<double, double>(mean, stddev);
}

void PartitionAlgo::save_partition(const std::string &path, size_t n, idx_t *parti)
{
    // pinrt path.c_str()
    printf("Saving partition to %s\n", path.c_str());
    FILE *fp = fopen(path.c_str(), "wb");
    if (fp == nullptr) {
        throw std::runtime_error("Failed to open partition file: " + path);
    }
    // for (size_t i = 0; i < n; i++) {
    //     fwrite(parti + i, sizeof(idx_t), 1, fp);
    // }
    fwrite(parti, sizeof(idx_t), n, fp);
    // for (size_t i = 0; i < n; i++) {
    //     fprintf(fp, "%lld\n", parti[i]);
    // }
    // fclose(fp);
}

idx_t* PartitionAlgo::load_partition(const std::string &file_name, size_t* nptr)
{
    FILE *fp = fopen(file_name.c_str(), "rb");
    if (fp == nullptr) {
        throw std::runtime_error("Failed to open partition file: " + file_name);
    }
    fseek(fp, 0, SEEK_END);
    size_t file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if(!nptr){
        nptr = new size_t;
    }
    *nptr = file_size / sizeof(idx_t);
    int n = *nptr;
    idx_t *parti = new idx_t[n];
    size_t read_count = fread(parti, sizeof(idx_t), n, fp);
    if (read_count != n) {
        fclose(fp);
        delete[] parti;
        throw std::runtime_error("Failed to read partition file: " + file_name);
    }
    return parti;
}

RandPartition::RandPartition(size_t d_, size_t nlist_, unsigned seed): PartitionAlgo(d_, nlist_)
{
    srand(seed);
}

RandPartition::~RandPartition()
{
}

void RandPartition::partition(size_t n, const float *data, idx_t *res, std::string sav)
{
    #pragma omp parallel for 
    for (size_t i = 0; i < n; i++) {
        res[i] = rand() % nlist;
    }
    if (!sav.empty()) {
        // save_partition(sav, n, res);
        save(sav);
    }
}

void RandPartition::rank(size_t n, const float *query, size_t k, idx_t *res)
{
    #pragma omp parallel for 
    for(int i=0;i<n;i++){
        std::vector<int> possible_values(nlist);
        std::iota(possible_values.begin(), possible_values.end(), 0); // Fills with 0, 1, ..., nlist-1
        // Shuffle the vector using the thread-local random engine
        std::random_shuffle(possible_values.begin(), possible_values.end());
        for(int j = 0; j < k; ++j) {
            res[i * k + j] = possible_values[j];
        }
    }
}

void RandPartition::save(std::string sav)
{
    ConfigWriter writer;
    writer.set("dim", d);
    writer.set("nlist", nlist);
    writer.set("method", "rand");
    writer.save(sav + "conf.json");
    printf("Random partition saved to %s\n", (sav + "conf.json").c_str());
}
