# Lesson 07 — Streams & Events
add_executable(07_streams_events streams_events.cu)
target_link_libraries(07_streams_events PRIVATE CUDA::cudart)
set_project_warnings(07_streams_events)

if(CUDA_TUTORIAL_BUILD_TESTS)
  add_executable(07_streams_events_test streams_events_test.cu)
  target_link_libraries(07_streams_events_test PRIVATE GTest::gtest_main CUDA::cudart)
  set_project_warnings(07_streams_events_test)
  gtest_discover_tests(07_streams_events_test)
endif()
