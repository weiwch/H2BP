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
using grpc::CompletionQueue;
using anns_service::Query;
using anns_service::ResultOfQuery;
using grpc::ClientAsyncResponseReader;
using anns_service::BatchQuery;
using anns_service::BatchResult;

// struct for keeping state and data information
struct AsyncClientCall {
    // Container for the data we expect from the server.
    ResultOfQuery reply;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;

    // Storage for the status of the RPC upon completion.
    Status status;

    std::unique_ptr<ClientAsyncResponseReader<ResultOfQuery>> response_reader;
};

// **NEW**: Struct for keeping state for a batch async query
struct AsyncBatchClientCall {
    BatchResult reply;
    ClientContext context;
    Status status;
    std::unique_ptr<ClientAsyncResponseReader<BatchResult>> response_reader;
};

class AnnsClientAsync {
public:
    explicit AnnsClientAsync(std::shared_ptr<Channel> channel, int nq)
        : stub_(AnnsService::NewStub(channel)) {
        call_list = new AsyncClientCall*[nq];
        batch_call_list = new AsyncBatchClientCall*[nq];
    }

    ~AnnsClientAsync() {
        // Note: The original code leaks call_list if AsyncCompleteRpc is not called.
        // In a real application, resource management should be more robust.
    }

    // Assembles and sends a single query.
    void Query(const float* queries, const int dim, const int k, const int tag) {
        anns_service::Query req;
        req.mutable_vec()->Add(queries, queries + dim);
        req.set_k(k);

        call_list[tag] = new AsyncClientCall;
        auto call = call_list[tag];

        call->response_reader =
            stub_->PrepareAsyncSearch(&call->context, req, &cq_);
        call->response_reader->StartCall();
        call->response_reader->Finish(&call->reply, &call->status, (void*)((size_t)tag));
    }

    // Waits for all single queries to complete.
    void AsyncCompleteRpc(int nns, int nq, int *res) {
        void* got_tag;
        bool ok = false;
        int cnt = 0;

        while (cq_.Next(&got_tag, &ok)) {
            size_t tag = (size_t)got_tag;
            AsyncClientCall* call = call_list[tag];

            CHECK(ok);

            if (call->status.ok()) {
                // Corrected memcpy: use sizeof(uint32_t) or sizeof(int) for IDs
                // and ensure the destination pointer is correct.
                memcpy(res + (nns * tag), call->reply.ids().data(), nns * sizeof(uint32_t));
            } else {
                std::cout << "RPC failed for tag " << tag << std::endl;
            }

            delete call;
            if (++cnt == nq) {
                break;
            }
        }
        delete[] call_list;
        call_list = nullptr; // Avoid double delete
    }

    void BatchQuery(const float* queries, const int n_queries, const int dim, const int k, const int tag) {
        // printf("BatchQuery %d\n", tag);
        // fflush(stdout);
        anns_service::BatchQuery req;
        req.mutable_vec()->Add(queries, queries + (n_queries * dim));
        req.set_k(k);
        req.set_n(n_queries);

        // Use a unique tag for the batch call to distinguish it from single calls.
        // We'll use a pointer to the call object itself as the tag.
        batch_call_list[tag] = new AsyncBatchClientCall;

        auto batch_call = batch_call_list[tag];
        batch_call->response_reader = stub_->PrepareAsyncBatchSearch(&batch_call->context, req, &cq_);
        batch_call->response_reader->StartCall();
        batch_call->response_reader->Finish(&batch_call->reply, &batch_call->status, (void*)((size_t)tag));
    }

    void AsyncCompleteBatchRpc(int nns, int n_queries, int *res) {
        void* got_tag;
        bool ok = false;
        int cnt = 0;
        // printf("AsyncCompleteBatchRpc start\n");
        // fflush(stdout);
        // Block until the batch result is available.
        while (cq_.Next(&got_tag, &ok)) {
            // printf("Got tag %p\n", got_tag);
            // fflush(stdout);
            size_t tag = (size_t)got_tag;
            printf("Processing batch call %zu\n", tag);
            fflush(stdout);
            AsyncBatchClientCall* call = batch_call_list[tag];
            CHECK(ok);

            if (call->status.ok()) {
                int current_offset = 0;
                // The reply contains a list of results, one for each query in the batch.
                // We iterate through them and copy the IDs to our final result array.
                
                auto ofs =  tag * nns * call->reply.results_size();
                for (const auto& single_result : call->reply.results()) {
                    // if(cnt<=2){
                    //     for(int i=0;i<nns;++i){
                    //         printf("%u ",single_result.ids(i));
                    //     }
                    //     puts("");
                    //     fflush(stdout);
                    // }
                    memcpy(res + ofs + current_offset, single_result.ids().data(), nns * sizeof(uint32_t));
                    current_offset += nns;
                }
            } else {
                std::cout << "Batch RPC failed" << std::endl;
            }
            // Deallocate the call object once we're done.
            delete call;
            if (++cnt == n_queries) {
                printf("All batch queries processed\n");
                fflush(stdout);
                break;
            }
        }
        delete[] batch_call_list;
        batch_call_list = nullptr; // Avoid double delete
    }

private:
    AsyncClientCall** call_list = nullptr;
    AsyncBatchClientCall** batch_call_list = nullptr;
    std::unique_ptr<AnnsService::Stub> stub_;
    CompletionQueue cq_;
};
