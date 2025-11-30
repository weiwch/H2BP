#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "dynamic_benchmark.h"
#include "utils.h"
#include <thread>
#include <filesystem>

std::string get_dataset_name(const std::string& path_str) {
    std::filesystem::path path(path_str);
    std::string filename = path.stem().string(); // "sift_base"
    // size_t pos = filename.find_last_of("_");
    // if (pos != std::string::npos) {
    //     return filename.substr(0, pos);
    // }
    return filename;
}

void print_results(const std::vector<DynamicBenchmark::ResultPair>& results, int k) {
    std::cout << std::fixed << std::setprecision(6);
    int count = 0;
    for (const auto& pair : results) {
        if (count++ >= k) break;
        std::cout << "  Index: " << std::setw(6) << pair.second << ", Distance: " << pair.first << "\n";
    }
}

void save_results_ivecs(const DynamicBenchmark& benchmark, const std::string& output_path) {
    std::ofstream os(output_path, std::ios::binary);
    if (!os) {
        throw std::runtime_error("Cannot open file for writing: " + output_path);
    }

    auto nq = static_cast<int32_t>(benchmark.get_num_queries());
    os.write(reinterpret_cast<const char*>(&nq), sizeof(nq));
    auto k_val = static_cast<int32_t>(benchmark.get_k());
    os.write(reinterpret_cast<const char*>(&k_val), sizeof(k_val));

    for (size_t i = 0; i < nq; ++i) {
        auto results = benchmark.get_results(i);

        std::vector<int32_t> indices;
        indices.reserve(k_val);
        for (const auto& p : results) {
            indices.push_back(p.second);
        }
        while (indices.size() < k_val) {
            indices.push_back(-1);
        }
        os.write(reinterpret_cast<const char*>(indices.data()), k_val * sizeof(int32_t));
    }
}


int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] << " <base_path> <query_path> <gt_path> <k> <base_ratio_percent> <chunk_size>\n";
        std::cerr << "Example: " << argv[0] << " sift_base.fvecs sift_query.fvecs sift_gt.ivecs 100 80 10000\n";
        return 1;
    }

    const std::string base_path = argv[1];
    const std::string query_path = argv[2];
    const std::string gt_path = argv[3];
    const int k = std::stoi(argv[4]);
    const float base_ratio = std::stof(argv[5]);
    const size_t chunk_size = std::stoul(argv[6]);

    if (chunk_size == 0) {
        std::cerr << "Error: chunk_size must be a positive integer." << std::endl;
        return 1;
    }

    try {
        DynamicBenchmark benchmark(base_path, query_path, gt_path, k, base_ratio);

        std::filesystem::path gt_fs_path(gt_path);
        std::string output_dir = gt_fs_path.parent_path().string();
        std::string dataset_name = get_dataset_name(base_path);
        
        std::stringstream ss_fname;
        ss_fname << output_dir + "/" + dataset_name << "_" << static_cast<int>(base_ratio) << ".fbin";
        fbin_write(ss_fname.str().c_str(), benchmark.dataset_.d, benchmark.get_num_base(), benchmark.dataset_.xb);
        std::cout << "Base vectors saved to: " << ss_fname.str() << std::endl;
        
        // return 0;

        benchmark.initialize();
        
        size_t inserted_count = 0;
        std::stringstream ss;
        ss << output_dir << "/" << dataset_name << "_" << (benchmark.get_num_base() + benchmark.get_num_inserted()) / 1000 << "k.ibin";
        std::string output_path = ss.str();
        std::cout << "Saving initial results (0 insertions) to: " << output_path << std::endl;
        save_results_ivecs(benchmark, output_path);

        while (benchmark.get_num_remaining() > 0) {
            benchmark.insert_n(chunk_size);
            
            inserted_count = benchmark.get_num_inserted();
            
            ss.str("");
            ss.clear();
            ss << output_dir << "/" << dataset_name << "_" << (benchmark.get_num_base() + benchmark.get_num_inserted()) / 1000 << "k.ibin";
            output_path = ss.str();
            
            std::cout << "Saving results after " << inserted_count << " total insertions to: " << output_path << std::endl;
            save_results_ivecs(benchmark, output_path);
        }

        std::cout << "\nBenchmark finished. All intermediate results have been saved." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}