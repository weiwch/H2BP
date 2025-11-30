#pragma once 
#include "partition/partition.h"
#include "partition/partition_kmeans.h"
#include "partition/partition_hybrid.h"
#include "partition/partition_graph.h"
#include "partition/partition_graphp.h"
#include "config.h"
#include <memory>

std::shared_ptr<PartitionAlgo> load_from_file(const std::string& path){
    std::shared_ptr<PartitionAlgo> partition;
    const std::string config_path = path + "conf.json";
    auto cfg = Config(config_path);
    std::string partition_method = cfg.get<std::string>("method");
    if(partition_method == "kmeans"){
        partition = KmeansPartition::load(path);
    }else if(partition_method == "rand"){
        partition = std::make_shared<RandPartition>(cfg.get<int>("dim"), cfg.get<int>("nlist"), 0);
    }else if(partition_method == "hb"){
        partition = HybridTreePartition::load(path);
    }else if(partition_method == "hb_t"){
        int nsublist = cfg.get<int>("nsublist");
        auto p = std::make_shared<HybridTreePartition>(cfg.get<int>("dim"), cfg.get<int>("nlist"), nsublist);
        p->routing_method = 0;
        p->partition_method = 0;
        partition = p;
    }else if(partition_method == "graph"){
        partition = GraphPartition::load(path);
        
    }else if(partition_method == "gp"){
        partition = GraphPartition2::load(path);
        
    }else{
        // throw error
        fprintf(stderr, "Unknown partition method: %s\n", partition_method.c_str());
        abort();
    }
    return partition;
}