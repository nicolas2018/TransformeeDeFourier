find_package( OpenCV REQUIRED )
add_executable( run main.cpp )
target_link_libraries( run ${OpenCV_LIBS} )
