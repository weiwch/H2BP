
file(GLOB HDR_FILES *.h)
file(GLOB SRC_FILES *.cpp)

file(GLOB SUB_HDR_FILES "partition/*.h")
list(APPEND HDR_FILES "${SUB_HDR_FILES}")
file(GLOB SUB_SRC_FILES "partition/*.cpp")
list(APPEND SRC_FILES "${SUB_SRC_FILES}")


function(add_subdirectory_executable target_subdir)
    set(SUBDIR_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${target_subdir})

    if(NOT IS_DIRECTORY ${SUBDIR_PATH})
        message(WARNING "${target_subdir} does not exist in ${CMAKE_CURRENT_SOURCE_DIR}")
        return()
    endif()

    file(GLOB_RECURSE SUB_HDR_FILES
        ${SUBDIR_PATH}
        "${SUBDIR_PATH}/*.h"
    )
    file(GLOB_RECURSE SUB_SRC_FILES
        ${SUBDIR_PATH}
        "${SUBDIR_PATH}/*.cpp"
    )

    if(NOT SUB_SRC_FILES)
        message(WARNING "no cpp file in ${target_subdir}")
        return()
    endif()
    list(APPEND SUB_HDR_FILES ${HDR_FILES})
    message(${HDR_FILES})
    list(APPEND SUB_SRC_FILES ${SRC_FILES})
    message(${SRC_FILES})
    add_executable(${target_subdir} ${SUB_SRC_FILES} ${SUB_HDR_FILES})
    target_include_directories(${target_subdir} PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/faiss
    )
    target_link_libraries(${target_subdir} PUBLIC MKL::MKL OpenMP::OpenMP_CXX MPI::MPI_CXX absl::flags absl::flags_parse absl::log
    absl::log_initialize
    absl::log_sink
    absl::check
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/faiss/_libfaiss_stage/lib/libfaiss_avx512.so
    )

    set_target_properties(${target_subdir} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../../binary/)

    target_sources(${target_subdir} PUBLIC 
        datasets.cpp
        ../thirdparty/gp-ann/src/defs.cpp
		../thirdparty/gp-ann/src/dist.cpp
		../thirdparty/gp-ann/src/kmeans.cpp
		../thirdparty/gp-ann/src/points_io.cpp
		../thirdparty/gp-ann/src/metis_io.cpp
        ../thirdparty/gp-ann/src/partitioning.cpp ../thirdparty/gp-ann/src/overlapping_partitioning.cpp ../thirdparty/gp-ann/src/kmeans_tree_router.cpp)
    
    target_link_libraries(${target_subdir} PUBLIC kaminpar_shm parlay)
endfunction()

find_package(absl REQUIRED) # https://abseil.io/docs/cpp/tools/cmake-installs
# Or build the absl in your repo by `add_subdirectory(abseil-cpp)`

add_subdirectory_executable(indexer)
add_subdirectory_executable(graph_indexer)
add_subdirectory_executable(hnsw_indexer)
add_subdirectory_executable(queryer)
add_subdirectory_executable(optimizer)
add_subdirectory_executable(optimizer_bp)
add_subdirectory_executable(optimizer_gd)
add_subdirectory_executable(check)
add_subdirectory_executable(bench)

# if(ENABLE_GRPC_SERVER)
# For gRPC
include(../cmake/common.cmake)
# Proto file
get_filename_component(hw_proto "../protos/anns_service.proto" ABSOLUTE)
get_filename_component(hw_proto_path "${hw_proto}" PATH)
# Generated sources
set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/anns_service.pb.cc")
set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/anns_service.pb.h")
set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/anns_service.grpc.pb.cc")
set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/anns_service.grpc.pb.h")
add_custom_command(
      OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${hw_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${hw_proto}"
      DEPENDS "${hw_proto}")
# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")
# hw_grpc_proto
add_library(hw_grpc_proto
  ${hw_grpc_srcs}
  ${hw_grpc_hdrs}
  ${hw_proto_srcs}
  ${hw_proto_hdrs})
target_link_libraries(hw_grpc_proto
  absl::check
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})

add_subdirectory_executable(grpc_server)
add_subdirectory_executable(client)
# add_subdirectory_executable(client2)
target_link_libraries(grpc_server PUBLIC hw_grpc_proto)
target_link_libraries(client PUBLIC hw_grpc_proto)
# target_link_libraries(client2 PUBLIC hw_grpc_proto)

# endif()
