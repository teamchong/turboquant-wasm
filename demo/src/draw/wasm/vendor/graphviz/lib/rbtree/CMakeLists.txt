add_library(rbtree STATIC
  # Header files
  red_black_tree.h

  # Source files
  red_black_tree.c
)

include_directories(rbtree PRIVATE
  ..
)
