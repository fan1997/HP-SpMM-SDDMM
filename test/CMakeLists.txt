# Get all the test files
SET(TEST_SOURCES 
)

# Build an executable file for all test files
set(BUILD_TEST_MAIN ON)
if (${BUILD_TEST_MAIN})
  add_executable(main main.cpp ${TEST_SOURCES})
  target_link_libraries(main spmm)
endif()

set(BUILD_TEST_MAIN ON)
if (${BUILD_TEST_MAIN})
  add_executable(test_all test_all.cpp ${TEST_SOURCES})
  target_link_libraries(test_all spmm)
endif()


# Build separated executable files for each test files
foreach(test_file ${TEST_SOURCES})
  # Get name of the source testing file to be the name of executable files
  get_filename_component(test_name ${test_file} NAME_WE)
  add_executable(${test_name} main.cpp ${test_file})
  target_link_libraries(${test_name} spmm)
endforeach()