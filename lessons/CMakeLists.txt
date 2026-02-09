# =============================================================================
# lessons/CMakeLists.txt — Include every lesson sub-directory
# =============================================================================

add_subdirectory(01_device_query)
add_subdirectory(02_hello_kernel)
add_subdirectory(03_error_handling)
add_subdirectory(04_memory_model)
add_subdirectory(05_thread_hierarchy)
add_subdirectory(06_memory_transfers)
add_subdirectory(07_streams_events)
add_subdirectory(08_reduction)
add_subdirectory(09_scan)
add_subdirectory(10_transpose)
add_subdirectory(11_matmul)
add_subdirectory(12_dense_layer)
add_subdirectory(13_activations)
add_subdirectory(14_conv2d)
add_subdirectory(15_pooling)
add_subdirectory(16_loss)
add_subdirectory(17_training)
add_subdirectory(18_inference)
add_subdirectory(20_minibatch_training)
add_subdirectory(21_mixed_precision)
add_subdirectory(22_performance_measurement)

if(CUDA_TUTORIAL_USE_CUDNN)
  add_subdirectory(19_cublas_cudnn)
endif()
