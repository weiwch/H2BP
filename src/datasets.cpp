#include "datasets.h"

Dataset::Dataset(const std::string &data_path, const std::string &query_path, const std::string &gt_path, size_t bin_format) : xb(nullptr), xq(nullptr), gt(nullptr), d(0), nb(0), nq(0), gt_k(0), data_path_(data_path), query_path_(query_path), gt_path_(gt_path)
{
}

float *Dataset::read_float(std::string path, size_t *d_out, size_t *n_out)
{
    std::string suffix("bin");
    float *x;
    if (std::equal(suffix.rbegin(), suffix.rend(), path.rbegin())) { //endswith 
        printf("Reading %s in binary format...\n", path.c_str());
        x = fbin_read(path.c_str(), d_out, n_out);
    } else {
        printf("Reading %s in fvecs format...\n", path.c_str());
        x = fvecs_read(path.c_str(), d_out, n_out);
    }
    return x;
}

int *Dataset::read_int(std::string path, size_t *d_out, size_t *n_out)
{
    std::string suffix("bin");
    int *x;
    if (std::equal(suffix.rbegin(), suffix.rend(), path.rbegin())) { //endswith 
        printf("Reading %s in binary format...\n", path.c_str());
        x = ibin_read(path.c_str(), d_out, n_out);
    } else {
        printf("Reading %s in fvecs format...\n", path.c_str());
        x = ivecs_read(path.c_str(), d_out, n_out);
    }
    return x;
}

void Dataset::clear_data()
{
    delete[] xb;
    xb = nullptr;
}

void Dataset::clear_query()
{
    delete[] xq;
    xq = nullptr;
    delete[] gt;
    gt = nullptr;
}

Dataset::~Dataset()
{
    clear_data();
    clear_query();
}


void Dataset::read_data()
{
    if(xb){
        printf("Have already read! do nothing");
        return;
    }
    size_t read_d;
    xb = read_float(data_path_, &read_d, &nb);
    if(d && read_d !=d){
        throw std::invalid_argument("Dim of base vectors is not consistent");
    }else{
        d = read_d;
    }
}

void Dataset::read_query()
{
    if(xq){
        printf("Have already read! do nothing");
        return;
    }
    size_t read_d;
    xq = read_float(query_path_, &read_d, &nq);
    if(d && read_d !=d){
        throw std::invalid_argument("Dim of query vectors is not consistent");
    }else{
        d = read_d;
    }
    size_t gt_n;
    gt = read_int(gt_path_, &gt_k, &gt_n);
    if(gt_n != nq){
        throw std::invalid_argument("# of queries and labels are not consistent");
    }
}
