#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "anns_service.grpc.pb.h"
#include "partition/partition.h"
#include "partition/partition_loader.h"
#include <grpc/support/log.h>
#include <grpcpp/grpcpp.h>

using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerCompletionQueue;
using grpc::ServerContext;
using grpc::Status;
using anns_service::AnnsService;
using anns_service::Query;
using anns_service::ResultOfQuery;

class GrpcAsyncServer final {
 public:
  GrpcAsyncServer(const std::string &working_dir, std::shared_ptr<PartitionAlgo> parti_algo);

  ~GrpcAsyncServer();
    
  // There is no shutdown handling in this code.
  void Run(const uint16_t &port, const int num_threads);

 private:
  
  // This can be run in multiple threads if needed.
  void HandleRpcs();

  std::unique_ptr<ServerCompletionQueue> cq_;
  AnnsService::AsyncService service_;
  std::unique_ptr<Server> server_;
  std::string working_dir_;
  std::shared_ptr<PartitionAlgo> parti_algo_;
};

