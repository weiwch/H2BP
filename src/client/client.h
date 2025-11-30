#pragma once
#include "utils.h"
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <omp.h>
#include <vector>
#include "absl/log/check.h"

#include <grpcpp/grpcpp.h>
#include "anns_service.grpc.pb.h"

// 64-bit int
typedef int64_t idx_t;

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using anns_service::AnnsService;
using anns_service::Query;
using anns_service::ResultOfQuery;
using grpc::ClientAsyncResponseReader;
using anns_service::BatchQuery;
using anns_service::BatchResult;
using anns_service::BatchInsQuery;
using anns_service::ResultOfInsert;


class AnnsClient {
public:
    AnnsClient(std::shared_ptr<Channel> channel)
        : stub_(AnnsService::NewStub(channel)) {}
    // Synchronous method for single query
    void Query(const float* queries, const int dim, const int k, ResultOfQuery* res) {
        anns_service::Query req;
        // More efficient way to add repeated fields
        req.mutable_vec()->Add(queries, queries + dim);
        req.set_k(k);

        ClientContext context;
        Status status = stub_->Search(&context, req, res);
        if (!status.ok()) {
            std::cout << status.error_code() << ": " << status.error_message()
                      << std::endl;
        }
    }

    // Synchronous method for batch queries
    void BatchQuery(const float* queries, const int n_queries, const int dim, const int k, BatchResult* res) {
        anns_service::BatchQuery req;
        // Add all query vectors at once from the flat array
        req.mutable_vec()->Add(queries, queries + (n_queries * dim));
        req.set_k(k);
        req.set_n(n_queries);

        ClientContext context;
        // Call the BatchSearch RPC
        Status status = stub_->BatchSearch(&context, req, res);

        if (!status.ok()) {
            std::cout << status.error_code() << ": " << status.error_message()
                      << std::endl;
        }
    }

    void Insert(const float* vectors, const int n_vectors, const int dim, const int start_idx, ResultOfInsert* res) {
        anns_service::BatchInsQuery req;
        // Add all vectors at once from the flat array
        req.mutable_vec()->Add(vectors, vectors + (n_vectors * dim));
        req.set_n(n_vectors);
        req.set_start_idx(start_idx);

        ClientContext context;
        // Call the Insert RPC
        Status status = stub_->Insert(&context, req, res);

        if (!status.ok()) {
            std::cout << status.error_code() << ": " << status.error_message()
                      << std::endl;
        }
    }

private:
    std::unique_ptr<AnnsService::Stub> stub_;
};
