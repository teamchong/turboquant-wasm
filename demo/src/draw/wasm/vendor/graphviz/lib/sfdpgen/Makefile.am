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

noinst_HEADERS = spring_electrical.h \
	sparse_solve.h post_process.h \
	stress_model.h \
	Multilevel.h sfdp.h

if WITH_SFDP
noinst_LTLIBRARIES = libsfdpgen_C.la
endif

libsfdpgen_C_la_SOURCES = sfdpinit.c spring_electrical.c \
	sparse_solve.c post_process.c \
	stress_model.c \
	Multilevel.c
