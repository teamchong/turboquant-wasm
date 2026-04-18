add_definitions(-DNEATOGEN_EXPORTS)

add_library(twopigen STATIC
  # Header files
  circle.h

  # Source files
  circle.c
  twopiinit.c
)

target_include_directories(twopigen PRIVATE
  ..
  ../cdt
  ../cgraph
  ../common
  ../gvc
  ../pack
  ../pathplan
)

target_link_libraries(twopigen PRIVATE
  cgraph
)
