#pragma once

#include <grpc/support/log.h>
#include <grpcpp/grpcpp.h>
#include "anns_service.grpc.pb.h"
#include "partition/partition.h"
#include <mpi.h>
#include <atomic>

using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerCompletionQueue;
using grpc::ServerContext;
using grpc::Status;
using anns_service::AnnsService;

class CallData {
public:
  // The means of communication with the gRPC runtime for an asynchronous
  // server.
  AnnsService::AsyncService* service_;
  // The producer-consumer queue where for asynchronous server notifications.
  ServerCompletionQueue* cq_;

  std::shared_ptr<PartitionAlgo> parti_algo_;

  // CallData(AnnsService::AsyncService* service, ServerCompletionQueue* cq, PartitioningAlgo* parti_algo)
  //       : service_(service), cq_(cq), parti_algo_(parti_algo) {}

};

class Call {
 public:
  virtual void Proceed() = 0;

  static std::atomic<int> tag_idx_;
 protected:
 // Let's implement a tiny state machine with the following states.
  enum CallStatus { CREATE, PROCESS, FINISH };
};

class SearchCall final : public Call {
public:
    explicit SearchCall(CallData *data) : data_(data), responder_(&ctx_), status_(CREATE) {
        // Invoke the serving logic right away.
        Proceed();
    }
    void Proceed();
private:
  void ProcessSearchRequest();
  CallData *data_;
  // Context for the rpc, allowing to tweak aspects of it such as the use
  // of compression, authentication, as well as to send metadata back to the
  // client.
  ServerContext ctx_;

  // What we get from the client.
  anns_service::Query request_;
  // What we send back to the client.
  anns_service::ResultOfQuery reply_;

  // The means to get back to the client.
  ServerAsyncResponseWriter<anns_service::ResultOfQuery> responder_;

  CallStatus status_;  // The current serving state.
};


class InsertCall final : public Call {
public:
    explicit InsertCall(CallData *data) : data_(data), responder_(&ctx_), status_(CREATE) {
        // Invoke the serving logic right away.
        Proceed();
    }
    void Proceed();
private:
  void ProcessInsertRequest();
  CallData *data_;
  // Context for the rpc, allowing to tweak aspects of it such as the use
  // of compression, authentication, as well as to send metadata back to the
  // client.
  ServerContext ctx_;

  // What we get from the client.
  anns_service::BatchInsQuery request_;
  // What we send back to the client.
  anns_service::ResultOfInsert reply_;

  // The means to get back to the client.
  ServerAsyncResponseWriter<anns_service::ResultOfInsert> responder_;

  CallStatus status_;  // The current serving state.
};


class BatchSearchCall final : public Call {
public:
    explicit BatchSearchCall(CallData *data) : data_(data), responder_(&ctx_), status_(CREATE) {
        // Invoke the serving logic right away.
        Proceed();
    }
    void Proceed();

private:
  void ProcessBatchSearchRequest();
  CallData *data_;
  // Context for the rpc, allowing to tweak aspects of it such as the use
  // of compression, authentication, as well as to send metadata back to the
  // client.
  ServerContext ctx_;

  // What we get from the client.
  anns_service::BatchQuery request_;
  // What we send back to the client.
  anns_service::BatchResult reply_;

  // The means to get back to the client.
  ServerAsyncResponseWriter<anns_service::BatchResult> responder_;

  CallStatus status_;  // The current serving state.
};