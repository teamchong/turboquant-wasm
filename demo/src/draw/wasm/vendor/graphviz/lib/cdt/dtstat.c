#include "config.h"

#include <cdt/dthdr.h>
#include <stdlib.h>

/*	Get statistics of a dictionary
**
**	Written by Kiem-Phong Vo (5/25/96)
*/

static void dttstat(Dtstat_t *ds, Dtlink_t *root, size_t depth, size_t *level) {
  if (root->left)
    dttstat(ds, root->left, depth + 1, level);
  if (root->right)
    dttstat(ds, root->right, depth + 1, level);
  if (depth > ds->dt_n)
    ds->dt_n = depth;
  if (level)
    level[depth] += 1;
}

static void dthstat(Dtdata_t data, Dtstat_t *ds, size_t *count) {
  for (int h = data.ntab - 1; h >= 0; --h) {
    size_t n = 0;
    for (Dtlink_t *t = data.htab[h]; t; t = t->right)
      n += 1;
    if (count)
      count[n] += 1;
    else if (n > 0) {
      ds->dt_n += 1;
      if (n > ds->dt_max)
        ds->dt_max = n;
    }
  }
}

int dtstat(Dt_t *dt, Dtstat_t *ds, int all) {
  static size_t *Count;

  UNFLATTEN(dt);

  ds->dt_n = ds->dt_max = 0;
  ds->dt_count = NULL;
  ds->dt_size = dtsize(dt);
  ds->dt_meth = dt->data.type & DT_METHODS;

  if (!all)
    return 0;

  if (dt->data.type & DT_SET) {
    dthstat(dt->data, ds, NULL);
    free(Count);
    if (!(Count = calloc(ds->dt_max + 1, sizeof(size_t))))
      return -1;
    dthstat(dt->data, ds, Count);
  } else if (dt->data.type & (DT_OSET | DT_OBAG)) {
    if (dt->data.here) {
      dttstat(ds, dt->data.here, 0, NULL);
      free(Count);
      if (!(Count = calloc(ds->dt_n + 1, sizeof(size_t))))
        return -1;
      dttstat(ds, dt->data.here, 0, Count);
      for (size_t i = 0; i <= ds->dt_n; ++i)
        if (Count[i] > ds->dt_max)
          ds->dt_max = Count[i];
    }
  }
  ds->dt_count = Count;

  return 0;
}
