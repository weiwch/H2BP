#include "dynamic_benchmark.h"
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <omp.h>


DynamicBenchmark::DynamicBenchmark(
    const std::string& base_path,
    const std::string& query_path,
    const std::string& gt_path,
    int k,
    int base_ratio_percent,
    Metric metric
) : dataset_(base_path, query_path, gt_path), k_(k), metric_(metric), current_dynamic_offset_(0) {
    if (k_ <= 0) throw std::invalid_argument("k must be positive.");
    if (base_ratio_percent < 0 || base_ratio_percent > 100) {
        throw std::invalid_argument("Base ratio must be between 0 and 100.");
    }

    // Load data
    std::cout << "Loading dataset...\n";
    dataset_.read_data();
    dataset_.read_query();
    std::cout << "Dataset loaded: nb=" << dataset_.nb << ", nq=" << dataset_.nq << ", d=" << dataset_.d << "\n\n";

    // Split data
    num_base_ = static_cast<size_t>(dataset_.nb * base_ratio_percent / 100);
    num_dynamic_ = dataset_.nb - num_base_;
    base_vectors_ = dataset_.xb;
    dynamic_vectors_ = dataset_.xb + num_base_ * dataset_.d;
    std::cout << "Data split: " << num_base_ << " initial vectors, " << num_dynamic_ << " dynamic vectors for insertion.\n\n";
    if(metric_ == Metric::L2){
        std::cout << "Using L2 distance metric.\n\n";
        space_ = new hnswlib::L2Space(dataset_.d);
    }else{
        std::cout << "Using Inner Product distance metric.\n\n";
        space_ = new hnswlib::InnerProductSpace(dataset_.d);
    }
    // Initialize heaps with base data
    // initialize();
}

void DynamicBenchmark::initialize() {
    std::cout << "Initializing " << dataset_.nq << " heaps with " << num_base_ << " base vectors...\n";
    auto start = time_now();
    top_k_heaps_.resize(dataset_.nq);

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < dataset_.nq; ++i) {
        top_k_heaps_[i].reserve(k_);
        float* query_vec = dataset_.xq + i * dataset_.d;
        for(size_t j = 0; j < k_; ++j){
            float* vec = base_vectors_ + j * dataset_.d;
            float dist = space_->get_dist_func()(query_vec, vec, space_->get_dist_func_param());
            top_k_heaps_[i].push_back({dist, (int)j});
        }
        std::make_heap(top_k_heaps_[i].begin(), top_k_heaps_[i].end());
        for (size_t j = 0; j < num_base_; ++j) {
            float* vec = base_vectors_ + j * dataset_.d;
            float dist = space_->get_dist_func()(query_vec, vec, space_->get_dist_func_param());
            
            if (dist < top_k_heaps_[i].front().first) {
                std::pop_heap(top_k_heaps_[i].begin(), top_k_heaps_[i].end());
                top_k_heaps_[i].back() = {dist, (int)j};
                std::push_heap(top_k_heaps_[i].begin(), top_k_heaps_[i].end());
            }
        }
        // If there were fewer than k base vectors, ensure it's still a heap
        if (top_k_heaps_[i].size() < k_ && !top_k_heaps_[i].empty()) {
            std::make_heap(top_k_heaps_[i].begin(), top_k_heaps_[i].end());
        }
    }
    std::cout << "Initialization complete, cost " <<  elapsed_ms(start)/1000 << " sec\n";
}

void DynamicBenchmark::insert_n(size_t num_to_insert) {
    if (num_to_insert <= 0) return;
    
    size_t end_offset = std::min(current_dynamic_offset_ + num_to_insert, num_dynamic_);
    if (current_dynamic_offset_ >= end_offset) {
        std::cout << "No more dynamic vectors to insert.\n";
        return;
    }

    std::cout << "Inserting " << (end_offset - current_dynamic_offset_) << " vectors...\n";

    
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = current_dynamic_offset_; i < end_offset; ++i) {
        float* dynamic_vec = dynamic_vectors_ + i * dataset_.d;
        int dynamic_vec_idx = num_base_ + i;

        for (size_t j = 0; j < dataset_.nq; ++j) {

            float* query_vec = dataset_.xq + j * dataset_.d;
            float dist = space_->get_dist_func()(query_vec, dynamic_vec, space_->get_dist_func_param());
            // `front()` of a max-heap is the largest element.
            if (dist < top_k_heaps_[j].front().first) {
                // This sequence is the standard way to replace the top element in a vector-based heap
                std::pop_heap(top_k_heaps_[j].begin(), top_k_heaps_[j].end());
                top_k_heaps_[j].back() = {dist, dynamic_vec_idx};
                std::push_heap(top_k_heaps_[j].begin(), top_k_heaps_[j].end());
            }
        }
    }
    current_dynamic_offset_ = end_offset;
}


std::vector<DynamicBenchmark::ResultPair> DynamicBenchmark::get_results(size_t query_id) const {
    if (query_id >= dataset_.nq) {
        throw std::out_of_range("Query ID is out of range.");
    }
    
    // Create a copy to avoid modifying the internal heap structure.
    auto results_copy = top_k_heaps_[query_id];

    // sort_heap converts the max-heap into a sorted range (ascending).
    std::sort_heap(results_copy.begin(), results_copy.end());

    return results_copy;
}