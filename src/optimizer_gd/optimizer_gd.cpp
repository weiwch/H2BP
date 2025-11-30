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

struct Item {
    double recall;
    double cost;
    int ef;
    bool operator<(const Item& other) const {
        return recall < other.recall;
    }
};

vector<vector<Item>> gp;
int ngroup;

const int bpc = 10;
double bp[] = {0.995, 0.99, 0.98, 0.97, 0.96, 0.95, 0.93, 0.9, 0.8, 0.7};

ABSL_FLAG(std::string, load, "", "Path to the configuration and data directory.");

void solve_with_greedy(FILE* fp) {
    cout << "\n--- Starting Greedy Algorithm ---" << endl;

    for (int i = 0; i < ngroup; ++i) {
        gp[i].push_back({0.0, 0.0, -1});
        sort(gp[i].begin(), gp[i].end());
    }
    vector<int> current_indices(ngroup, 0);
    double total_recall = 0.0;
    double total_cost = 0.0;
    // int used_nodes = 0;
    for (int gidx = bpc - 1; gidx >= 0; -- gidx) {
        double target_recall = bp[gidx];
        
        int best_group_to_upgrade;
        while (total_recall < target_recall) {
            best_group_to_upgrade = -1;
            double max_efficiency = -1.0;

            for (int g = 0; g < ngroup; ++g) {
                if (current_indices[g] + 1 < gp[g].size()) {
                    const auto& current_item = gp[g][current_indices[g]];
                    const auto& next_item = gp[g][current_indices[g] + 1];

                    double delta_recall = next_item.recall - current_item.recall;
                    double delta_cost = next_item.cost - current_item.cost;
                    
                    if (delta_cost <= 1e-9) { 
                        if (delta_recall > 1e-9) {
                           max_efficiency = DBL_MAX;
                           best_group_to_upgrade = g;
                        }
                        continue;
                    }

                    double efficiency = delta_recall / delta_cost;
                    if (efficiency > max_efficiency) {
                        max_efficiency = efficiency;
                        best_group_to_upgrade = g;
                    }
                }
            }
            // if(current_indices[best_group_to_upgrade] == 0){
            //     used_nodes++;
            // }
            if (best_group_to_upgrade == -1) {
                break;
            }

            const auto& prev_item = gp[best_group_to_upgrade][current_indices[best_group_to_upgrade]];
            current_indices[best_group_to_upgrade]++;
            const auto& next_item = gp[best_group_to_upgrade][current_indices[best_group_to_upgrade]];
            
            total_recall += (next_item.recall - prev_item.recall);
            total_cost += (next_item.cost - prev_item.cost);
        }
        
        printf("Target recall: %.4f -> Achieved recall: %.4f, cost: %.4f\n",
               target_recall, total_recall, total_cost);

        if(total_recall < target_recall) {
            continue;
        }

        // if(used_nodes > 1 && current_indices[best_group_to_upgrade] == 0){
        //     // use the node next time
        // }

        double recall_except_last = 0.0;
        for (int i = 0; i < ngroup; ++i) {
            if(i != best_group_to_upgrade){
                recall_except_last += gp[i][current_indices[i]].recall;
            }
        }
        auto last_item = gp[best_group_to_upgrade][current_indices[best_group_to_upgrade]];
        int delta_ef = last_item.ef * (target_recall-recall_except_last) / last_item.recall;
        for (int i = 0; i < ngroup; ++i) {
            if(i != best_group_to_upgrade){
                fprintf(fp, "%d ", gp[i][current_indices[i]].ef);
            }
            else{
                fprintf(fp, "%d ", delta_ef);
            }
        }
        fprintf(fp, "%.6f\n", target_recall);
    }
}


int main(int argc, char** argv) {
    absl::ParseCommandLine(argc, argv);

    if (absl::GetFlag(FLAGS_load).empty()) {
        cerr << "Require an index path. Use --load <path>" << endl;
        return -1;
    }

    std::string prefix = absl::GetFlag(FLAGS_load) + "/";

    Document doc;
    const std::string confPath = prefix + "conf.json";
    ifstream confFile(confPath);
    if (!confFile.is_open()) {
        cerr << "Error opening file: " << confPath << endl;
        return -1;
    }
    stringstream buffer;
    buffer << confFile.rdbuf();
    string jsonContent = buffer.str();
    doc.Parse(jsonContent.c_str());
    if (doc.HasParseError()) {
        cerr << "Error parsing JSON..." << endl;
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
            gp[idx_].push_back({recall_, cost_, ef_});
        }
    }
    fclose(fpr);
    
    cout << "Data loaded. Number of groups: " << ngroup << endl;

    std::string out_filename = prefix + "ef_file.csv";
    FILE* fp = fopen(out_filename.c_str(), "w");
    if (!fp) {
        perror(("Could not open for writing: " + out_filename).c_str());
        return -1;
    }

    solve_with_greedy(fp);

    fclose(fp);

    vector<string> lines;
    ifstream infile(out_filename);
    string line;
    if (!infile.is_open()) {
        cerr << "Error re-opening file for reading: " << out_filename << endl;
        return -1;
    }
    while (getline(infile, line)) {
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    infile.close();

    std::reverse(lines.begin(), lines.end());

    FILE* fp_out = fopen(out_filename.c_str(), "w");
    if (!fp_out) {
        perror(("Could not open for final writing: " + out_filename).c_str());
        return -1;
    }
    for (const auto& l : lines) {
        fprintf(fp_out, "%s\n", l.c_str());
    }
    fclose(fp_out);

    cout << "File content reversed." << endl;

    return 0;
}