# Compares the H.264 MV grid of the export with FFmpeg's own MV export
add_executable(h264-mv-check h264-mv-check.cpp)
target_link_libraries(h264-mv-check PRIVATE videoparser)
target_include_directories(h264-mv-check PRIVATE ${FFMPEG_DIR})

# Compares FrameInfo::maps with the export files
add_executable(maps-check maps-check.cpp)
target_link_libraries(maps-check PRIVATE videoparser)
target_include_directories(maps-check PRIVATE ${FFMPEG_DIR})
