#include "grpc_async_call.h"
#include "absl/log/check.h"
#include "ann_search.h"
#include "ann_insert.h"
#include <atomic>

std::atomic<int> Call::tag_idx_{0};

void SearchCall::ProcessSearchRequest()
{
    const float *vec = request_.vec().data();
    int res_check_nns = request_.k();
    idx_t* Ir = new idx_t[res_check_nns * 1];
    auto algo = data_->parti_algo_;
    int base_idx = tag_idx_.fetch_add(1, std::memory_order_relaxed);
    ann_search<false>(vec, 1, algo->d, Ir,res_check_nns, res_check_nns, algo->n_server, algo->nlist, algo->nsearch, algo->ief, algo.get(), nullptr, nullptr, base_idx);
    for(int i=0;i<res_check_nns;++i){
        reply_.add_ids(Ir[i]);
    }
    delete[] Ir;
}

void SearchCall::Proceed()
{
    if (status_ == CREATE) {
    // Make this instance progress to the PROCESS state.
    status_ = PROCESS;

    // As part of the initial CREATE state, we *request* that the system
    // start processing SayHello requests. In this request, "this" acts are
    // the tag uniquely identifying the request (so that different CallData
    // instances can serve different requests concurrently), in this case
    // the memory address of this CallData instance.
    data_->service_->RequestSearch(&ctx_, &request_, &responder_, data_->cq_, data_->cq_,
                           this);
  } else if (status_ == PROCESS) {
    // Spawn a new CallData instance to serve new clients while we process
    // the one for this CallData. The instance will deallocate itself as
    // part of its FINISH state.
    new SearchCall(data_);

    // The actual processing.
    ProcessSearchRequest();
    // And we are done! Let the gRPC runtime know we've finished, using the
    // memory address of this instance as the uniquely identifying tag for
    // the event.
    status_ = FINISH;
    responder_.Finish(reply_, Status::OK, this);
  } else {
    CHECK(status_ == FINISH);
    // Once in the FINISH state, deallocate ourselves (Call).
    delete this;
  }
}

void BatchSearchCall::ProcessBatchSearchRequest()
{
    const float *vec = request_.vec().data();
    int res_check_nns = request_.k();
    int n_query = request_.n();
    idx_t* Ir = new idx_t[res_check_nns * n_query];
    auto algo = data_->parti_algo_;
    int base_idx = tag_idx_.fetch_add(n_query, std::memory_order_relaxed);
    
    ann_search<false>(vec, n_query, algo->d, Ir,res_check_nns, res_check_nns, algo->n_server, algo->nlist, algo->nsearch, algo->ief, algo.get(), nullptr, nullptr, base_idx);  
   
    for (int i = 0; i < n_query; ++i) {
      auto res = reply_.add_results();
      for(int j=0;j<res_check_nns;++j){
        res->add_ids(Ir[i * res_check_nns + j]);
      }
    }
    delete[] Ir;
}

void BatchSearchCall::Proceed()
{
      if (status_ == CREATE) {
    // Make this instance progress to the PROCESS state.
    status_ = PROCESS;

    // As part of the initial CREATE state, we *request* that the system
    // start processing SayHello requests. In this request, "this" acts are
    // the tag uniquely identifying the request (so that different CallData
    // instances can serve different requests concurrently), in this case
    // the memory address of this CallData instance.
    data_->service_->RequestBatchSearch(&ctx_, &request_, &responder_, data_->cq_, data_->cq_,
                           this);
  } else if (status_ == PROCESS) {
    // Spawn a new CallData instance to serve new clients while we process
    // the one for this CallData. The instance will deallocate itself as
    // part of its FINISH state.
    new BatchSearchCall(data_);

    // The actual processing.
    ProcessBatchSearchRequest();
    // And we are done! Let the gRPC runtime know we've finished, using the
    // memory address of this instance as the uniquely identifying tag for
    // the event.
    status_ = FINISH;
    responder_.Finish(reply_, Status::OK, this);
  } else {
    CHECK(status_ == FINISH);
    // Once in the FINISH state, deallocate ourselves (Call).
    delete this;
  }
}


void InsertCall::Proceed()
{
    if (status_ == CREATE) {
    // Make this instance progress to the PROCESS state.
    status_ = PROCESS;

    // As part of the initial CREATE state, we *request* that the system
    // start processing SayHello requests. In this request, "this" acts are
    // the tag uniquely identifying the request (so that different CallData
    // instances can serve different requests concurrently), in this case
    // the memory address of this CallData instance.
    data_->service_->RequestInsert(&ctx_, &request_, &responder_, data_->cq_, data_->cq_,
                           this);
  } else if (status_ == PROCESS) {
    // Spawn a new CallData instance to serve new clients while we process
    // the one for this CallData. The instance will deallocate itself as
    // part of its FINISH state.
    new InsertCall(data_);

    // The actual processing.
    ProcessInsertRequest();
    // And we are done! Let the gRPC runtime know we've finished, using the
    // memory address of this instance as the uniquely identifying tag for
    // the event.
    status_ = FINISH;
    responder_.Finish(reply_, Status::OK, this);
  } else {
    CHECK(status_ == FINISH);
    // Once in the FINISH state, deallocate ourselves (Call).
    delete this;
  }
}

void InsertCall::ProcessInsertRequest()
{
    const float *vec = request_.vec().data();
    int n_query = request_.n();
    int start_idx = request_.start_idx();
    auto algo = data_->parti_algo_;
    int base_idx = tag_idx_.fetch_add(n_query, std::memory_order_relaxed);
    int status = ann_insert(algo.get(), vec, n_query, algo->d, algo->n_server, algo->nlist, start_idx, base_idx);
    reply_.set_status(status);
}
