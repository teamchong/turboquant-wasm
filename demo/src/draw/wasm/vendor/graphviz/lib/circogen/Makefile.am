## Process this file with automake to produce Makefile.in

AM_CPPFLAGS = \
	-I$(top_srcdir)/lib \
	-I$(top_srcdir)/lib/common \
	-I$(top_srcdir)/lib/gvc \
	-I$(top_srcdir)/lib/pack \
	-I$(top_srcdir)/lib/pathplan \
	-I$(top_srcdir)/lib/cgraph \
	-I$(top_srcdir)/lib/cdt

if WITH_WIN32
AM_CFLAGS = -DNEATOGEN_EXPORTS=1
endif

noinst_HEADERS = block.h blockpath.h blocktree.h circo.h \
	circpos.h circular.h edgelist.h nodelist.h
noinst_LTLIBRARIES = libcircogen_C.la

libcircogen_C_la_SOURCES = circularinit.c nodelist.c block.c edgelist.c \
	circular.c blocktree.c blockpath.c \
	circpos.c
