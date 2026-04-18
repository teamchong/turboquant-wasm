add_library(sparse STATIC
  # Header files
  clustering.h
  color_palette.h
  colorutil.h
  DotIO.h
  general.h
  mq.h
  QuadTree.h
  SparseMatrix.h

  # Source files
  clustering.c
  color_palette.c
  colorutil.c
  DotIO.c
  general.c
  mq.c
  QuadTree.c
  SparseMatrix.c
)

target_include_directories(sparse PRIVATE
  ..
  ../cdt
  ../cgraph
  ../common
)
