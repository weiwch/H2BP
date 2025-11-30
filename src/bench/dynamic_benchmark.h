#pragma once

#include <string>
#include <vector>
#include <utility>
#include "datasets.h"
#include "hnswlib/hnswlib.h"
#include "hnswlib/space_l2.h"
#include "hnswlib/space_ip.h"

// Enum to select distance metric
enum class Metric { L2, IP };

class DynamicBenchmark {
public:
    // Alias for the result pairs (distance, index)
    using ResultPair = std::pair<float, int>;

    /**
     * @brief Constructor to set up the benchmark.
     * @param base_path Path to the base vector data.
     * @param query_path Path to the query vector data.
     * @param gt_path Path to the ground truth data.
     * @param k The number of nearest neighbors to track.
     * @param base_ratio_percent The percentage of base vectors to use for initial population.
     * @param metric The distance metric to use (L2 or IP).
     */
    DynamicBenchmark(
        const std::string& base_path,
        const std::string& query_path,
        const std::string& gt_path,
        int k,
        int base_ratio_percent,
        Metric metric = Metric::L2
    );

    /**
     * @brief Inserts and processes the next 'num_to_insert' dynamic vectors.
     * @param num_to_insert The number of vectors to process from the dynamic set.
     */
    void insert_n(size_t num_to_insert);

    /**
     * @brief Retrieves the current top-k results for a specific query.
     * @param query_id The index of the query.
     * @return A sorted vector of (distance, index) pairs, from best to worst.
     */
    std::vector<ResultPair> get_results(size_t query_id) const;

    
    void initialize();
    // --- Status Methods ---
    size_t get_num_inserted() const { return current_dynamic_offset_; }
    size_t get_num_remaining() const { return num_dynamic_ - current_dynamic_offset_; }
    size_t get_num_queries() const { return dataset_.nq; }
    size_t get_num_base() const { return num_base_; }
    int get_k() const { return k_; }

    Dataset dataset_;
private:
    int k_;
    Metric metric_;

    // Pointers to data segments
    float* base_vectors_;
    float* dynamic_vectors_;
    size_t num_base_;
    size_t num_dynamic_;

    // Progress tracker
    size_t current_dynamic_offset_;
    hnswlib::SpaceInterface<float>* space_;

    // The data structure for the heaps.
    // Each inner vector is managed as a max-heap using std::*_heap algorithms.
    std::vector<std::vector<ResultPair>> top_k_heaps_;
};
