## Process this file with automake to produce Makefile.in

AM_CPPFLAGS = \
	-I$(top_srcdir)/lib \
	-I$(top_srcdir)/lib/common \
	-I$(top_srcdir)/lib/cgraph \
	-I$(top_srcdir)/lib/cdt

noinst_HEADERS = SparseMatrix.h general.h DotIO.h \
	colorutil.h color_palette.h mq.h clustering.h QuadTree.h

noinst_LTLIBRARIES = libsparse_C.la

libsparse_C_la_SOURCES = SparseMatrix.c general.c DotIO.c \
	colorutil.c color_palette.c mq.c clustering.c QuadTree.c
