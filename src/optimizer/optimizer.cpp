#include <bits/stdc++.h>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "rapidjson/document.h"
using namespace std;
using namespace rapidjson;

typedef tuple<double, double, int> itm;
vector<vector<itm> > gp;
int bpc = 10; // break point count
int ngroup;
double bp[] = {0.995, 0.99, 0.98, 0.97, 0.96, 0.95, 0.93, 0.9, 0.8, 0.7}; // break point
// int gidx = bpc - 1;
int gidx = 0;
vector<int> sel_now;
vector<int> sel_ans;
double min_cst = 2e18;

void dfs(int i, double rec, double cst){
    if(i==ngroup){
        // for(int i=0;i<ngroup;++i){    
        //     printf("%d ",sel_now[i]);
        // }
        // printf("cst \n", cst);
        if(rec>bp[gidx] && cst < min_cst){
            min_cst = cst;
            sel_ans = sel_now;
            printf("%lf/%lf, cost : %lf\n", rec, bp[0], cst);
            for(int i=0;i<ngroup;++i){    
                printf("%d ",sel_now[i]);
            }
            printf("cst \n", cst);
            for(int i=0;i<ngroup;++i){
                
                if(sel_ans[i] >=0){
                    auto it = gp[i][sel_ans[i]];
                    printf("%d : %d, ef %d recall %lf cost %lf\n",i, sel_ans[i], get<2>(it), get<0>(it), get<1>(it));
                }
            }
            printf("\n\n");
        }
        return;
    }
    sel_now[i] = -1;
    dfs(i+1, rec, cst);
    // sel_now[i] = -1;
    for(int j = 0;j<gp[i].size();++j){
        sel_now[i] = j;
        dfs(i+1, rec + get<0>(gp[i][j]), cst + get<1>(gp[i][j]));
        sel_now[i] = -1;
    }
}

ABSL_FLAG(std::string, load, "", "");

int main(int argc, char** argv){

    absl::ParseCommandLine(argc, argv);

    if(strcmp(absl::GetFlag(FLAGS_load).c_str(), "") == 0){
        printf("Require a index path. Use --load <path>\n");
        return 0;
    }
    std::string prefix = absl::GetFlag(FLAGS_load) + "/";
    Document doc;
    const std::string filePath = prefix + "conf.json";
    std::ifstream file(filePath);
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string jsonContent = buffer.str();
    doc.Parse(jsonContent.c_str());
    if (doc.HasParseError()) {
        std::cerr << "Error parsing JSON: " << doc.GetParseError() << " at offset " << doc.GetErrorOffset() << std::endl;
        return -1;
    }
    ngroup = doc["nlist"].GetUint();
    // scanf("%d", &ngroup);
    gp.resize(ngroup);
    sel_now.resize(ngroup);
    int idx_, ef_;
    double recall_, cost_;
    // while (scanf("%d%d%lf%lf", &idx_, &ef_, &recall_, &cost_) != EOF){
    //     gp[idx_].push_back(make_tuple(recall_, cost_, ef_));
    // }
    string read_filename = prefix + "all_res.csv";
    FILE* fpr = fopen(read_filename.c_str(), "r");
    if(!fpr){
        cout<<read_filename<<endl;
        perror("Can not open file");
        return -1;
    }
    while (fscanf(fpr, "%d%d%lf%lf", &idx_, &ef_, &recall_, &cost_) != EOF){
        gp[idx_].push_back(make_tuple(recall_, cost_, ef_));
    }
    for(int i=0;i<ngroup;++i){
        
        printf("server %d-th:\n", i);
        for(auto it : gp[i]){
            
            printf("ef %d: recall %lf cost %lf\n", get<2>(it), get<0>(it), get<1>(it));
        }
    }
    std::string filename = prefix + "ef_file.csv";
    FILE* fp = fopen(filename.c_str(), "w");
    // fprintf(fp, "%d\n", bpc);
    printf("FILE Opened\n");
    fflush(stdout);
    // for (; gidx >= 0; -- gidx){
    for (; gidx < bpc; ++ gidx){
        for(int i=0;i<ngroup;++i){
            sel_now[i] = -1;
            min_cst = 2e18;
        }
        dfs(0, 0.0, 0.0);
        printf("%lf min cost %lf\n", bp[gidx], min_cst);
        if(min_cst>2e9){
            continue;
        }
        for(int i=0;i<ngroup;++i){
            if(sel_ans[i] >= 0){
                auto it = gp[i][sel_ans[i]];
                printf("%d, ef %d recall %lf cost %lf\n",sel_ans[i], get<2>(it), get<0>(it), get<1>(it));
                fprintf(fp, "%d ", get<2>(it));
            }else{
                fprintf(fp, "-1 ");
            }
            
        }
        fprintf(fp, "%lf\n", bp[gidx]);
        // break; 
    }
    fclose(fp);
    return 0;
}
