#pragma once

#include <string>
#include <cstddef>
#include "utils.h"
#include <stdexcept>

class Dataset {
public:
    float* xb;       // Pointer to base vectors
    float* xq;       // Pointer to query vectors
    int* gt;         // Pointer to ground truth
    size_t d;       // Data dimensionality
    size_t nb;       // Number of base vectors
    size_t nq;       // Number of query vectors
    size_t gt_k;     // Number of neighbors per query in ground truth
    std::string data_path_;
    std::string query_path_;
    std::string gt_path_;
    // Constructor: Loads data from the given file paths
    Dataset(const std::string& data_path, const std::string& query_path = "", const std::string& gt_path = "", size_t bin_format = 0);

    // Destructor: Automatically clears allocated memory
    ~Dataset();

    // Clears all allocated memory
    void clear_data();
    void clear_query();

    // Accessor methods
    void read_data();
    void read_query();

private:
    static float* read_float(std::string path, size_t* d_out, size_t* n_out);
    static int* read_int(std::string path, size_t* d_out, size_t* n_out);
};
