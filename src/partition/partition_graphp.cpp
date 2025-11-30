#include "partition_graphp.h"
#include "partitioning.h"
#include "metis_io.h"
#include <string>
#include "hnswlib/hnswlib.h"
#include <set>
#include "config.h"
#include <faiss/index_io.h>
#include "utils.h"

GraphPartition2::GraphPartition2(size_t d_, size_t nlist_, int budget_): PartitionAlgo(d_, nlist_)
{
    budget = budget_;
    routing_index_options.budget = budget;
    routing_index_options.search_budget = budget;
    trained = false;
}

GraphPartition2::~GraphPartition2()
{
    routing_index_options.budget = budget;
    routing_index_options.search_budget = budget;
    trained = false;
}

void GraphPartition2::train(size_t n, const float *data)
{
    // do nothing
}

void GraphPartition2::partition(size_t n, const float *data, idx_t *res, std::string sav)
{

    auto points = PointSet();
    points.n = n;
    points.d = d;
    // copy data to points.coordinates
    // points.coordinates.resize(n * d);
    // std::memcpy(points.coordinates.data(), data, n * d * sizeof(float));
    points.coordinates.assign(data, data + (n * d));

    auto partition = GraphPartitioning(points, nlist, 0.05, false, sav + index_name);
    
    for(int i=0;i<partition.size();i++){
        res[i] = partition[i];
    }
    // points.Drop();
    
    auto clusters = ConvertPartitionToClusters(partition);
    // printf("Routing index built 111.\n");
    // fflush(stdout);
    trained = true;
    router.Train(points, clusters, routing_index_options);
    // build routing index
    // printf("Routing index built.\n");
    fflush(stdout);
    points.Drop();
    save_partition(sav + "partition.idx", n, res);
    save(sav);
}

void GraphPartition2::rank(size_t n, const float *query, size_t k, idx_t *res)
{

    Timer routing_timer;
    std::vector<std::vector<int>> buckets_to_probe_by_query(n);
    double time_routing;
    float* dq = const_cast<float*>(query);
    // parlay::execute_with_scheduler(std::min<size_t>(32, parlay::num_workers()), [&] {
    //     routing_timer.Start();
    //     parlay::parallel_for(0, n, [&](size_t i) {
    //         buckets_to_probe_by_query[i] = router.Query(dq + i*d, routing_index_options.search_budget);
    //     });
    //     time_routing = routing_timer.Stop();
    // });
    for(int i=0;i<n;i++)
        buckets_to_probe_by_query[i] = router.Query(dq + i*d, routing_index_options.search_budget);
    
    std::cout << "KMTR Routing took " << time_routing << " s overall, and " << time_routing / n
                      << " s per query."<< std::endl;
    for(int i=0;i<n;i++){
        // printf("qsz %d: %d", i, buckets_to_probe_by_query[i].size());
        for(int j=0;j<std::min<size_t>(k, buckets_to_probe_by_query[i].size());j++){
            res[i*k + j] = buckets_to_probe_by_query[i][j];
        }
    }
}

void GraphPartition2::save(std::string sav)
{
    // save index
    // faiss::write_index(&top_quantizer, (sav + index_name).c_str());
    ConfigWriter writer;
    writer.set("dim", d);
    writer.set("nlist", nlist);
    writer.set("budget", budget);
    writer.set("method", "gp");
    writer.set("datapath", datapath.c_str());
    writer.save(sav + "conf.json");
    
}

std::shared_ptr<GraphPartition2> GraphPartition2::load(std::string sav){
    const std::string filePath = sav + "conf.json";
    auto cfg = Config(filePath);
    auto p = std::make_shared<GraphPartition2>(cfg.get<int>("dim"), cfg.get<int>("nlist"), cfg.get<int>("budget"));
    // load partiton
    size_t n = 0;
    idx_t *parti = p->load_partition(sav + "partition.idx", &n);
    // covert idx_t array to Partition of GP-ANN
    Partition partition;
    for(int i=0;i<n;++i){
        partition.push_back(parti[i]);
    }
    auto clusters = ConvertPartitionToClusters(partition);

    p->trained = true;

    std::string datapath = cfg.get<std::string>("datapath");
    p->datapath = datapath;
    // load data points from datapath
    size_t dout, nout;
    float * data = fbin_read(datapath.c_str(), &dout, &nout);
    auto points = PointSet();
    points.n = nout;
    points.d = dout;
    points.coordinates.assign(data, data + (nout * dout));
    printf("START to reconstruct routing index from file %s with %zu points and %zu dim\n", datapath.c_str(), nout, dout);
    p->router.Train(points, clusters, p->routing_index_options);
    printf("GP-ANN LOAD OK!!!");
    return p;
}