/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

#include "config.h"

#include	<stdbool.h>
#include	<stddef.h>
#include	<stdio.h>
#include	<string.h>
#include	<unistd.h>

#ifdef ENABLE_LTDL
#include	<ltdl.h>
#endif

#include        <common/types.h>
#include        <gvc/gvc.h>
#include        <gvc/gvplugin.h>
#include        <gvc/gvcjob.h>
#include        <gvc/gvcint.h>
#include        <gvc/gvcproc.h>
#include        <gvc/gvio.h>

#include	<common/const.h>
#include <util/agxbuf.h>
#include <util/alloc.h>
#include <util/list.h>
#include <util/path.h>
#include <util/startswith.h>
#include <util/strcasecmp.h>
#include <util/strview.h>

/*
 * Define an apis array of name strings using an enumerated api_t as index.
 * The enumerated type is defined gvplugin.h.  The apis array is
 * initialized here by redefining ELEM and reinvoking APIS.
 */
#define ELEM(x) #x,
static char *api_names[] = { APIS };    /* "render", "layout", ... */

#undef ELEM

/* translate a string api name to its type, or -1 on error */
api_t gvplugin_api(const char *str)
{
    for (size_t api = 0; api < ARRAY_SIZE(api_names); api++) {
        if (strcmp(str, api_names[api]) == 0)
            return (api_t) api;
    }
    return -1;                  /* invalid api */
}

/* translate api_t into string name, or NULL */
char *gvplugin_api_name(api_t api)
{
    if (api >= ARRAY_SIZE(api_names))
        return NULL;
    return api_names[api];
}

/* install a plugin description into the list of available plugins
 * list is alpha sorted by type (not including :dependency), then
 * quality sorted within the type, then, if qualities are the same,
 * last install wins.
 */
bool gvplugin_install(GVC_t *gvc, api_t api, const char *typestr, int quality,
                      gvplugin_package_t *package,
                      gvplugin_installed_t *typeptr)
{
    /* duplicate typestr to later save in the plugin list */
    char *const t = strdup(typestr);
    if (t == NULL)
        return false;

    // find the current plugin
    const strview_t type = strview(typestr, ':');

    /* point to the beginning of the linked list of plugins for this api */
    gvplugin_available_t **pnext = &gvc->apis[api];

    /* keep alpha-sorted and insert new duplicates ahead of old */
    while (*pnext) {

        // find the next plugin
        const strview_t next_type = strview((*pnext)->typestr, ':');

        if (strview_cmp(type, next_type) <= 0)
            break;
        pnext = &(*pnext)->next;
    }

    /* keep quality sorted within type and insert new duplicates ahead of old */
    while (*pnext) {

        // find the next plugin
        const strview_t next_type = strview((*pnext)->typestr, ':');

        if (!strview_eq(type, next_type))
            break;
        if (quality >= (*pnext)->quality)
            break;
        pnext = &(*pnext)->next;
    }

    gvplugin_available_t *const plugin = gv_alloc(sizeof(gvplugin_available_t));
    plugin->next = *pnext;
    *pnext = plugin;
    plugin->typestr = t;
    plugin->quality = quality;
    plugin->package = package;
    plugin->typeptr = typeptr;  /* null if not loaded */

    return true;
}

/* Activate a plugin description in the list of available plugins.
 * This is used when a plugin-library loaded because of demand for
 * one of its plugins. It updates the available plugin data with
 * pointers into the loaded library.
 * NB the quality value is not replaced as it might have been
 * manually changed in the config file.
 */
static void gvplugin_activate(GVC_t * gvc, api_t api, const char *typestr,
                              const char *name, const char *plugin_path,
                              gvplugin_installed_t * typeptr)
{
    gvplugin_available_t *pnext;

    /* point to the beginning of the linked list of plugins for this api */
    pnext = gvc->apis[api];

    while (pnext) {
        if (strcasecmp(typestr, pnext->typestr) == 0
            && strcasecmp(name, pnext->package->name) == 0
            && pnext->package->path != 0
            && strcasecmp(plugin_path, pnext->package->path) == 0) {
            pnext->typeptr = typeptr;
            return;
        }
        pnext = pnext->next;
    }
}

gvplugin_library_t *gvplugin_library_load(GVC_t *gvc, const char *pathname) {
#ifdef ENABLE_LTDL
    lt_dlhandle hndl;
    lt_ptr ptr;
    char *s;
    size_t len;
    char *libdir;
    char *suffix = "_LTX_library";

    if (!gvc->common.demand_loading)
        return NULL;

    libdir = gvconfig_libdir(gvc);
    agxbuf fullpath = {0};
#ifdef _WIN32
    if (pathname[1] == ':') {
#else
    if (pathname[0] == '/') {
#endif
        agxbput(&fullpath, pathname);
    } else {
        agxbprint(&fullpath, "%s%c%s", libdir, PATH_SEPARATOR, pathname);
    }

    if (lt_dlinit()) {
        agerrorf("failed to init libltdl\n");
        agxbfree(&fullpath);
        return NULL;
    }
    char *p = agxbuse(&fullpath);
    hndl = lt_dlopen(p);
    if (!hndl) {
        if (access(p, R_OK) == 0) {
            agwarningf("Could not load \"%s\" - %s\n", p, "It was found, so perhaps one of its dependents was not.  Try ldd.");
        }
        else {
            agwarningf("Could not load \"%s\" - %s\n", p, lt_dlerror());
        }
        agxbfree(&fullpath);
        return NULL;
    }
    if (gvc->common.verbose >= 2)
        fprintf(stderr, "Loading %s\n", p);

    s = strrchr(p, PATH_SEPARATOR);
    len = strlen(s);
#if defined(_WIN32) && !defined(__MINGW32__) && !defined(__CYGWIN__)
    if (len < strlen("/gvplugin_x")) {
#else
    if (len < strlen("/libgvplugin_x")) {
#endif
        agerrorf("invalid plugin path \"%s\"\n", p);
        agxbfree(&fullpath);
        return NULL;
    }
    char *sym = gv_alloc(len + strlen(suffix) + 1);
#if defined(_WIN32) && !defined(__MINGW32__) && !defined(__CYGWIN__)
    strcpy(sym, s + 1);         /* strip leading "/"  */
#else
    strcpy(sym, s + 4);         /* strip leading "/lib" or "/cyg" */
#endif
#if defined(__CYGWIN__) || defined(__MINGW32__)
    s = strchr(sym, '-');       /* strip trailing "-1.dll" */
#else
    s = strchr(sym, '.');       /* strip trailing ".so.0" or ".dll" or ".sl" */
#endif
    strcpy(s, suffix);          /* append "_LTX_library" */

    ptr = lt_dlsym(hndl, sym);
    if (!ptr) {
        agerrorf("failed to resolve %s in %s\n", sym, p);
        free(sym);
        agxbfree(&fullpath);
        return NULL;
    }
    free(sym);
    agxbfree(&fullpath);
    return (gvplugin_library_t *)ptr;
#else
    (void)gvc;
    (void)pathname;

    agerrorf("dynamic loading not available\n");
    return NULL;
#endif
}


/* load a plugin of type=str
	the str can optionally contain one or more ":dependencies" 

	examples:
	        png
		png:cairo
        fully qualified:
		png:cairo:cairo
		png:cairo:gd
		png:gd:gd
      
*/
gvplugin_available_t *gvplugin_load(GVC_t *gvc, api_t api, const char *str,
                                    FILE *debug) {
    gvplugin_available_t *pnext, *rv;
    gvplugin_library_t *library;
    gvplugin_api_t *apis;
    gvplugin_installed_t *types;
    int i;
    api_t apidep;

    if (api == API_device || api == API_loadimage)
        /* api dependencies - FIXME - find better way to code these *s */
        apidep = API_render;
    else
        apidep = api;

    const strview_t reqtyp = strview(str, ':');

    strview_t reqdep = {0};

    strview_t reqpkg = {0};

    if (reqtyp.data[reqtyp.size] == ':') {
        reqdep = strview(reqtyp.data + reqtyp.size + strlen(":"), ':');
        if (reqdep.data[reqdep.size] == ':') {
            reqpkg = strview(reqdep.data + reqdep.size + strlen(":"), '\0');
        }
    }

    agxbuf diag = {0}; // diagnostic messages

    /* iterate the linked list of plugins for this api */
    for (pnext = gvc->apis[api]; pnext; pnext = pnext->next) {
        const strview_t typ = strview(pnext->typestr, ':');

        strview_t dep = {0};
        if (typ.data[typ.size] == ':') {
            dep = strview(typ.data + typ.size + strlen(":"), '\0');
        }

        if (!strview_eq(typ, reqtyp)) {
            agxbprint(&diag, "# type \"%.*s\" did not match \"%.*s\"\n",
                      (int)typ.size, typ.data, (int)reqtyp.size, reqtyp.data);
            continue;           /* types empty or mismatched */
        }
        if (dep.data && reqdep.data) {
            if (!strview_eq(dep, reqdep)) {
                agxbprint(&diag,
                          "# dependencies \"%.*s\" did not match \"%.*s\"\n",
                          (int)dep.size, dep.data, (int)reqdep.size,
                          reqdep.data);
                continue;           /* dependencies not empty, but mismatched */
            }
        }
        if (!reqpkg.data || strview_str_eq(reqpkg, pnext->package->name)) {
            // found with no packagename constraints, or with required matching packagename

            if (dep.data && apidep != api) // load dependency if needed, continue if can't find
                if (!gvplugin_load(gvc, apidep, dep.data, debug)) {
                    agxbprint(&diag,
                              "# plugin loading of dependency \"%.*s\" failed\n",
                              (int)dep.size, dep.data);
                    continue;
                }
            break;
        }
    }
    rv = pnext;

    if (rv && rv->typeptr == NULL) {
        library = gvplugin_library_load(gvc, rv->package->path);
        if (library) {

            /* Now activate the library with real type ptrs */
            for (apis = library->apis; (types = apis->types); apis++) {
                for (i = 0; types[i].type; i++) {
                    /* NB. quality is not checked or replaced
                     *   in case user has manually edited quality in config */
                    gvplugin_activate(gvc, apis->api, types[i].type, library->packagename, rv->package->path, &types[i]);
                }
            }
            if (gvc->common.verbose >= 1)
                fprintf(stderr, "Activated plugin library: %s\n", rv->package->path ? rv->package->path : "<builtin>");
        }
    }

    /* one last check for successful load */
    if (rv && rv->typeptr == NULL) {
        agxbprint(&diag, "# unsuccessful plugin load\n");
        rv = NULL;
    }

    if (rv && gvc->common.verbose >= 1)
        fprintf(stderr, "Using %s: %s:%s\n", api_names[api], rv->typestr, rv->package->name);

    if (debug != NULL) {
        fputs(agxbuse(&diag), debug);
    }
    agxbfree(&diag);

    gvc->api[api] = rv;
    return rv;
}

/* assemble a string list of available plugins 
 * non-re-entrant as character store is shared
 */
char *gvplugin_list(GVC_t * gvc, api_t api, const char *str)
{
    const gvplugin_available_t *pnext, *plugin;
    char *bp;
    bool new = true;
    static agxbuf xb;

    /* check for valid str */
    if (!str)
        return NULL;

    /* does str have a :path modifier? */
    const strview_t strv = strview(str, ':');

    /* point to the beginning of the linked list of plugins for this api */
    plugin = gvc->apis[api];

    if (strv.data[strv.size] == ':') { /* if str contains a ':', and if we find a match for the type,
                                          then just list the alternative paths for the plugin */
        for (pnext = plugin; pnext; pnext = pnext->next) {
            const strview_t type = strview(pnext->typestr, ':');
            // skip duplicates
            bool already_seen = false;
            for (const gvplugin_available_t *p = plugin; p != pnext;
                 p = p->next) {
                already_seen |= strcasecmp(pnext->typestr, p->typestr) == 0 &&
                  strcasecmp(pnext->package->name, p->package->name) == 0;
            }
            if (already_seen) {
                continue;
            }
            // list only the matching type, or all types if str is an empty
            // string or starts with ":"
            if (strv.size == 0 || strview_case_eq(strv, type)) {
                /* list each member of the matching type as "type:path" */
                agxbprint(&xb, " %s:%s", pnext->typestr, pnext->package->name);
                new = false;
            }
        }
    }
    if (new) {                  /* if the type was not found, or if str without ':',
                                   then just list available types */
        strview_t type_last = {0};
        for (pnext = plugin; pnext; pnext = pnext->next) {
            /* list only one instance of type */
            const strview_t type = strview(pnext->typestr, ':');
            if (!type_last.data || !strview_case_eq(type_last, type)) {
                /* list it as "type"  i.e. w/o ":path" */
                agxbprint(&xb, " %.*s", (int)type.size, type.data);
                new = false;
            }
            type_last = type;
        }
    }
    if (new)
        bp = "";
    else
        bp = agxbuse(&xb);
    return bp;
}

/* gvPluginList:
 * Return list of plugins of type kind.
 * The size of the list is stored in sz.
 * The caller is responsible for freeing the storage. This involves
 * freeing each item, then the list.
 * Returns NULL on error, or if there are no plugins.
 * In the former case, sz is unchanged; in the latter, sz = 0.
 */
char **gvPluginList(GVC_t *gvc, const char *kind, int *sz) {
    size_t api;
    const gvplugin_available_t *pnext, *plugin;
    LIST(char *) list = {0};

    if (!kind)
        return NULL;
    for (api = 0; api < ARRAY_SIZE(api_names); api++) {
        if (!strcasecmp(kind, api_names[api]))
            break;
    }
    if (api == ARRAY_SIZE(api_names)) {
        agerrorf("unrecognized api name \"%s\"\n", kind);
        return NULL;
    }

    /* point to the beginning of the linked list of plugins for this api */
    plugin = gvc->apis[api];
    strview_t typestr_last = {0};
    for (pnext = plugin; pnext; pnext = pnext->next) {
        /* list only one instance of type */
        strview_t q = strview(pnext->typestr, ':');
        if (!typestr_last.data || !strview_case_eq(typestr_last, q)) {
            LIST_APPEND(&list, strview_str(q));
        }
        typestr_last = q;
    }

    char **ret;
    size_t size;
    LIST_DETACH(&list, &ret, &size);
    *sz = (int)size;
    return ret;
}

void gvplugin_write_status(GVC_t * gvc)
{
    int api;

#ifdef ENABLE_LTDL
    if (gvc->common.demand_loading) {
        fprintf(stderr, "The plugin configuration file:\n\t%s\n", gvc->config_path);
        if (gvc->config_found)
            fprintf(stderr, "\t\twas successfully loaded.\n");
        else
            fprintf(stderr, "\t\twas not found or not usable. No on-demand plugins.\n");
    } else {
        fprintf(stderr, "Demand loading of plugins is disabled.\n");
    }
#endif

    for (api = 0; api < (int)ARRAY_SIZE(api_names); api++) {
        if (gvc->common.verbose >= 2)
            fprintf(stderr, "    %s\t: %s\n", api_names[api], gvplugin_list(gvc, api, ":"));
        else
            fprintf(stderr, "    %s\t: %s\n", api_names[api], gvplugin_list(gvc, api, "?"));
    }

}

Agraph_t *gvplugin_graph(GVC_t * gvc)
{
    char *p;

    Agraph_t *const g = agopen("G", Agdirected, NULL);
    agattr_text(g, AGRAPH, "label", "");
    agattr_text(g, AGRAPH, "rankdir", "");
    agattr_text(g, AGRAPH, "rank", "");
    agattr_text(g, AGRAPH, "ranksep", "");
    agattr_text(g, AGNODE, "label", NODENAME_ESC);
    agattr_text(g, AGNODE, "shape", "");
    agattr_text(g, AGNODE, "style", "");
    agattr_text(g, AGNODE, "width", "");
    agattr_text(g, AGEDGE, "style", "");

    Agsym_t *a = agfindgraphattr(g, "rankdir");
    agxset(g, a, "LR");

    a = agfindgraphattr(g, "ranksep");
    agxset(g, a, "2.5");

    a = agfindgraphattr(g, "label");
    agxset(g, a, "Plugins");

    agxbuf buf = {0};
    for (const gvplugin_package_t *package = gvc->packages; package; package = package->next) {
        Agnode_t *loadimage_n = NULL;
        Agnode_t *renderer_n = NULL;
        Agnode_t *device_n = NULL;
        Agnode_t *textlayout_n = NULL;
        Agnode_t *layout_n = NULL;
        bool neededge_loadimage = false;
        bool neededge_device = false;
        agxbprint(&buf, "cluster_%s", package->name);
        Agraph_t *const sg = agsubg(g, agxbuse(&buf), 1);
        a = agfindgraphattr(sg, "label");
        agxset(sg, a, package->name);
        for (size_t api = 0; api < ARRAY_SIZE(api_names); api++) {
            agxbprint(&buf, "%s_%s", package->name, api_names[api]);
            Agraph_t *const ssg = agsubg(sg, agxbuse(&buf), 1);
            a = agfindgraphattr(ssg, "rank");
            agxset(ssg, a, "same");
            for (const gvplugin_available_t *pnext = gvc->apis[api]; pnext; pnext = pnext->next) {
                if (pnext->package == package) {
                    char *const t = gv_strdup(pnext->typestr);
                    const char *q = t;
                    if ((p = strchr(q, ':')))
                        *p++ = '\0';
                    /* Now p = renderer, e.g. "gd"
                     * and q = device, e.g. "png"
                     * or  q = loadimage, e.g. "png" */
                    switch (api) {
                    case API_device:
                    case API_loadimage: {
			/* draw device as box - record last device in plugin  (if any) in device_n */
			/* draw loadimage as box - record last loadimage in plugin  (if any) in loadimage_n */

                        /* hack for aliases */
			const char *lq = q;
                        if (startswith(q, "jp")) {
                            q = "jpg";                /* canonical - for node name */
			    lq = "jpeg\\njpe\\njpg";  /* list - for label */
			}
                        else if (startswith(q, "tif")) {
                            q = "tif";
			    lq = "tiff\\ntif";
			}
                        else if (!strcmp(q, "x11") || !strcmp(q, "xlib")) {
                            q = "x11";
                            lq = "x11\\nxlib";
			}
                        else if (!strcmp(q, "dot") || !strcmp(q, "gv")) {
                            q = "gv";
                            lq = "gv\\ndot";
			}

                        agxbprint(&buf, "%s_%s_%s", package->name,
                                  api_names[api], q);
                        Agnode_t *const n = agnode(ssg, agxbuse(&buf), 1);
                        a = agfindnodeattr(g, "label");
                        agxset(n, a, lq);
                        a = agfindnodeattr(g, "width");
                        agxset(n, a, "1.0");
                        a = agfindnodeattr(g, "shape");
			if (api == API_device) {
                            agxset(n, a, "box");
                            device_n = n;
			}
                        else {
                            agxset(n, a, "box");
                            loadimage_n = n;
			}
                        if (!(p && *p)) {
                            Agnode_t *m = agfindnode(sg, "render_cg");
                            if (!m) {
                                m = agnode(sg, "render_cg", 1);
                                a = agfindgraphattr(g, "label");
                                agxset(m, a, "cg");
                            }
                            agedge(sg, m, n, NULL, 1);
                        }
                        break;
                    }
                    case API_render: {
			/* draw renderers as ellipses - record last renderer in plugin (if any) in renderer_n */
                        agxbprint(&buf, "%s_%s", api_names[api], q);
                        Agnode_t *const n = agnode(ssg, agxbuse(&buf), 1);
                        renderer_n = n;
                        a = agfindnodeattr(g, "label");
                        agxset(n, a, q);
                        break;
                    }
                    case API_textlayout: {
			/* draw textlayout  as invtriangle - record last textlayout in plugin  (if any) in textlayout_n */
			/* FIXME? only one textlayout is loaded. Why? */
                        agxbprint(&buf, "%s_%s", api_names[api], q);
                        Agnode_t *const n = agnode(ssg, agxbuse(&buf), 1);
                        textlayout_n = n;
                        a = agfindnodeattr(g, "shape");
                        agxset(n, a, "invtriangle");
                        a = agfindnodeattr(g, "label");
                        agxset(n, a, "T");
                        break;
                    }
                    case API_layout: {
			/* draw textlayout  as hexagon - record last layout in plugin  (if any) in layout_n */
                        agxbprint(&buf, "%s_%s", api_names[api], q);
                        Agnode_t *const n = agnode(ssg, agxbuse(&buf), 1);
                        layout_n = n;
                        a = agfindnodeattr(g, "shape");
                        agxset(n, a, "hexagon");
                        a = agfindnodeattr(g, "label");
                        agxset(n, a, q);
                        break;
                    }
                    default:
                        break;
                    }
                    free(t);
                }
            }
            // add some invisible nodes (if needed) and invisible edges to 
            //    improve layout of cluster
            if (api == API_loadimage && !loadimage_n) {
		neededge_loadimage = true;
                agxbprint(&buf, "%s_%s_invis", package->name, api_names[api]);
                Agnode_t *n = agnode(ssg, agxbuse(&buf), 1);
                loadimage_n = n;
                a = agfindnodeattr(g, "style");
                agxset(n, a, "invis");
                a = agfindnodeattr(g, "label");
                agxset(n, a, "");
                a = agfindnodeattr(g, "width");
                agxset(n, a, "1.0");

                agxbprint(&buf, "%s_%s_invis_src", package->name,
                          api_names[api]);
                n = agnode(g, agxbuse(&buf), 1);
                a = agfindnodeattr(g, "style");
                agxset(n, a, "invis");
                a = agfindnodeattr(g, "label");
                agxset(n, a, "");

                Agedge_t *const e = agedge(g, n, loadimage_n, NULL, 1);
                a = agfindedgeattr(g, "style");
                agxset(e, a, "invis");
	    }
            if (api == API_render && !renderer_n) {
		neededge_loadimage = true;
		neededge_device = true;
                agxbprint(&buf, "%s_%s_invis", package->name, api_names[api]);
                Agnode_t *n = agnode(ssg, agxbuse(&buf), 1);
                renderer_n = n;
                a = agfindnodeattr(g, "style");
                agxset(n, a, "invis");
                a = agfindnodeattr(g, "label");
                agxset(n, a, "");
	    }
            if (api == API_device && !device_n) {
		neededge_device = true;
                agxbprint(&buf, "%s_%s_invis", package->name, api_names[api]);
                Agnode_t *n = agnode(ssg, agxbuse(&buf), 1);
                device_n = n;
                a = agfindnodeattr(g, "style");
                agxset(n, a, "invis");
                a = agfindnodeattr(g, "label");
                agxset(n, a, "");
                a = agfindnodeattr(g, "width");
                agxset(n, a, "1.0");
	    }
        }
        if (neededge_loadimage) {
            Agedge_t *const e = agedge(sg, loadimage_n, renderer_n, NULL, 1);
            a = agfindedgeattr(g, "style");
            agxset(e, a, "invis");
        }
        if (neededge_device) {
            Agedge_t *const e = agedge(sg, renderer_n, device_n, NULL, 1);
            a = agfindedgeattr(g, "style");
            agxset(e, a, "invis");
        }
        if (textlayout_n) {
            Agedge_t *const e = agedge(sg, loadimage_n, textlayout_n, NULL, 1);
            a = agfindedgeattr(g, "style");
            agxset(e, a, "invis");
        }
        if (layout_n) {
            Agedge_t *const e = agedge(sg, loadimage_n, layout_n, NULL, 1);
            a = agfindedgeattr(g, "style");
            agxset(e, a, "invis");
        }
    }

    Agraph_t *const ssg = agsubg(g, "output_formats", 1);
    a = agfindgraphattr(ssg, "rank");
    agxset(ssg, a, "same");
    for (const gvplugin_package_t *package = gvc->packages; package; package = package->next) {
        for (size_t api = 0; api < ARRAY_SIZE(api_names); api++) {
            for (const gvplugin_available_t *pnext = gvc->apis[api]; pnext; pnext = pnext->next) {
                if (pnext->package == package) {
                    char *const t = gv_strdup(pnext->typestr);
                    const char *q = t;
                    if ((p = strchr(q, ':')))
                        *p++ = '\0';
                    /* Now p = renderer, e.g. "gd"
                     * and q = device, e.g. "png"
                     * or  q = imageloader, e.g. "png" */

 		    /* hack for aliases */
                    const char *lq = q;
                    if (startswith(q, "jp")) {
                        q = "jpg";                /* canonical - for node name */
                        lq = "jpeg\\njpe\\njpg";  /* list - for label */
                    }
                    else if (startswith(q, "tif")) {
                        q = "tif";
                        lq = "tiff\\ntif";
                    }
                    else if (!strcmp(q, "x11") || !strcmp(q, "xlib")) {
                        q = "x11";
                        lq = "x11\\nxlib";
                    }
                    else if (!strcmp(q, "dot") || !strcmp(q, "gv")) {
                        q = "gv";
                        lq = "gv\\ndot";
                    }

                    switch (api) {
                    case API_device: {
                        agxbprint(&buf, "%s_%s_%s", package->name,
                                  api_names[api], q);
                        Agnode_t *const n = agnode(g, agxbuse(&buf), 1);
                        agxbprint(&buf, "output_%s", q);
                        char *const output = agxbuse(&buf);
                        Agnode_t *m = agfindnode(ssg, output);
                        if (!m) {
                            m = agnode(ssg, output, 1);
                            a = agfindnodeattr(g, "label");
                            agxset(m, a, lq);
                            a = agfindnodeattr(g, "shape");
                            agxset(m, a, "note");
                        }
                        const Agedge_t *e = agfindedge(g, n, m);
                        if (!e)
                            agedge(g, n, m, NULL, 1);
                        if (p && *p) {
                            agxbprint(&buf, "render_%s", p);
                            char *const render = agxbuse(&buf);
                            m = agfindnode(ssg, render);
                            if (!m)
                                m = agnode(g, render, 1);
                            e = agfindedge(g, m, n);
                            if (!e)
                                agedge(g, m, n, NULL, 1);
                        }
                        break;
                    }
                    case API_loadimage: {
                        agxbprint(&buf, "%s_%s_%s", package->name,
                                  api_names[api], q);
                        Agnode_t *const n = agnode(g, agxbuse(&buf), 1);
                        agxbprint(&buf, "input_%s", q);
                        char *const input = agxbuse(&buf);
                        Agnode_t *m = agfindnode(g, input);
                        if (!m) {
                            m = agnode(g, input, 1);
                            a = agfindnodeattr(g, "label");
                            agxset(m, a, lq);
                            a = agfindnodeattr(g, "shape");
                            agxset(m, a, "note");
                        }
                        const Agedge_t *e = agfindedge(g, m, n);
                        if (!e)
                            agedge(g, m, n, NULL, 1);
                        agxbprint(&buf, "render_%s", p);
                        char *const render = agxbuse(&buf);
                        m = agfindnode(g, render);
                        if (!m)
                            m = agnode(g, render, 1);
                        e = agfindedge(g, n, m);
                        if (!e)
                            agedge(g, n, m, NULL, 1);
                        break;
                    }
                    default:
                        break;
                    }
                    free(t);
                }
            }
        }
    }

    agxbfree(&buf);
    return g;
}
