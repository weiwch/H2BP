#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <tuple>
#include <algorithm>
#include <cfloat> // For DBL_MAX

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "rapidjson/document.h"

using namespace std;
using namespace rapidjson;

// Item: recall, cost, ef
typedef tuple<double, double, int> itm;
vector<vector<itm>> gp;
int ngroup;
const int bpc = 10;
double bp[] = {0.995, 0.99, 0.98, 0.97, 0.96, 0.95, 0.93, 0.9, 0.8, 0.7};

const int PRECISION = 10000;

ABSL_FLAG(std::string, load, "", "Path to the configuration and data directory.");

int main(int argc, char** argv) {
    absl::ParseCommandLine(argc, argv);

    if (absl::GetFlag(FLAGS_load).empty()) {
        cerr << "Require an index path. Use --load <path>" << endl;
        return -1;
    }

    std::string prefix = absl::GetFlag(FLAGS_load) + "/";

    Document doc;
    const std::string confPath = prefix + "conf.json";
    std::ifstream confFile(confPath);
    if (!confFile.is_open()) {
        cerr << "Error opening file: " << confPath << endl;
        return -1;
    }
    std::stringstream buffer;
    buffer << confFile.rdbuf();
    std::string jsonContent = buffer.str();
    doc.Parse(jsonContent.c_str());
    if (doc.HasParseError()) {
        cerr << "Error parsing JSON: " << doc.GetParseError() << " at offset " << doc.GetErrorOffset() << endl;
        return -1;
    }
    ngroup = doc["nlist"].GetUint();
    gp.resize(ngroup);

    string read_filename = prefix + "all_res.csv";
    FILE* fpr = fopen(read_filename.c_str(), "r");
    if (!fpr) {
        perror(("Can not open file: " + read_filename).c_str());
        return -1;
    }
    int idx_, ef_;
    double recall_, cost_;
    while (fscanf(fpr, "%d%d%lf%lf", &idx_, &ef_, &recall_, &cost_) != EOF) {
        if (idx_ < ngroup) {
            gp[idx_].push_back(make_tuple(recall_, cost_, ef_));
        }
    }
    fclose(fpr);

    for (int i = 0; i < ngroup; ++i) {
        gp[i].push_back(make_tuple(0.0, 0.0, -1));
    }
    
    cout << "Data loaded. Number of groups: " << ngroup << endl;

    int max_scaled_recall = ngroup * PRECISION;

    // dp[g][r] the min cost to reach scaled_recall 'r' after processing g+1 groups 
    vector<vector<double>> dp(ngroup, vector<double>(max_scaled_recall + 1, DBL_MAX));

    // path[g][r] record which item was chosen for dp[g][r]
    vector<vector<int>> path(ngroup, vector<int>(max_scaled_recall + 1, -1));

    // init
    for (size_t j = 0; j < gp[0].size(); ++j) {
        auto& item = gp[0][j];
        int scaled_r = static_cast<int>(get<0>(item) * PRECISION);
        double current_cost = get<1>(item);
        
        if (current_cost < dp[0][scaled_r]) {
            dp[0][scaled_r] = current_cost;
            path[0][scaled_r] = j;
        }
    }

    for (int g = 1; g < ngroup; ++g) {
        for (int r = 0; r <= max_scaled_recall; ++r) {
            if (dp[g - 1][r] == DBL_MAX) {
                continue;
            }

            for (size_t j = 0; j < gp[g].size(); ++j) {
                auto& item = gp[g][j];
                int item_scaled_r = static_cast<int>(get<0>(item) * PRECISION);
                double item_cost = get<1>(item);

                int new_r = r + item_scaled_r;
                if (new_r <= max_scaled_recall) {
                    double new_cost = dp[g - 1][r] + item_cost;
                    if (new_cost < dp[g][new_r]) {
                        dp[g][new_r] = new_cost;
                        path[g][new_r] = j;
                    }
                }
            }
        }
    }
    
    cout << "DP calculation finished. Finding solutions for breakpoints." << endl;

    std::string out_filename = prefix + "ef_file.csv";
    FILE* fp = fopen(out_filename.c_str(), "w");
    if (!fp) {
        perror(("Could not open for writing: " + out_filename).c_str());
        return -1;
    }

    auto& final_dp_row = dp[ngroup - 1];

    for (int gidx = bpc - 1; gidx >= 0; -- gidx){
        double target_recall = bp[gidx];
        int target_scaled_recall = static_cast<int>(target_recall * PRECISION);

        double min_cst = DBL_MAX;
        int best_final_r = -1;

        for (int r = target_scaled_recall; r <= max_scaled_recall; ++r) {
            if (final_dp_row[r] < min_cst) {
                min_cst = final_dp_row[r];
                best_final_r = r;
            }
        }
        
        printf("Target recall: %.4f, ", target_recall);
        if (best_final_r == -1) {
            printf("No solution found.\n");
            continue;
        }
        printf("Found best solution with actual recall %.4f and min_cost %.4f\n", 
               (double)best_final_r / PRECISION, min_cst);
        
        vector<int> selected_efs(ngroup);
        int current_r = best_final_r;

        for (int g = ngroup - 1; g >= 0; --g) {
            int item_idx = path[g][current_r];
            if (item_idx == -1) {
                cerr << "Error in backtracking at group " << g << " with recall " << current_r << endl;
                exit(-1);
            }
            selected_efs[g] = get<2>(gp[g][item_idx]);
            
            int item_scaled_r = static_cast<int>(get<0>(gp[g][item_idx]) * PRECISION);
            current_r -= item_scaled_r;
        }

        for (int i = 0; i < ngroup; ++i) {
            fprintf(fp, "%d ", selected_efs[i]);
        }
        fprintf(fp, "%.6f\n", target_recall);
    }

    fclose(fp);
    cout << "Results written to " << out_filename << endl;

    return 0;
}