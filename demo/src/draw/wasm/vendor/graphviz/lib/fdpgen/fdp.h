/* Minimal fdp.h — only the struct definition needed by globals.c */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct fdpParms_s {
    int useGrid;
    int useNew;
    int numIters;
    int unscaled;
    double C;
    double Tfact;
    double K;
    double T0;
};
typedef struct fdpParms_s fdpParms_t;

#ifdef __cplusplus
}
#endif
