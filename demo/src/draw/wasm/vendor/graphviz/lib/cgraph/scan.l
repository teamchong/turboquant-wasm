/**
 * @file
 * @ingroup cgraph_core
 */
/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/


/* requires flex (i.e. not lex)  */

  /* By default, Flex emits a lexer using symbols prefixed with "yy". Graphviz
   * contains multiple Flex-generated lexers, so we alter this prefix to avoid
   * symbol clashes.
   */
%option prefix="aag"

   /* Generate a reentrant parser with no global variables.
    * All state will be contained in an opaque structure.
    */
%option reentrant bison-bridge

   /* The generated structure will have space for user data.
    */
%option extra-type="struct aagextra_s *"

  /* Avoid generating an unused input function. See
     https://westes.github.io/flex/manual/Scanner-Options.html
   */
%option noinput

%{
#include <assert.h>
#include <grammar.h>
#include <cgraph/cghdr.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <util/agxbuf.h>
#include <util/gv_ctype.h>
#include <util/startswith.h>
// #define YY_BUF_SIZE 128000
#define GRAPH_EOF_TOKEN		'@'		/* lex class must be defined below */
	/* this is a workaround for linux flex */

#define YYSTYPE AAGSTYPE

/* By default, Flex calls isatty() to determine whether the input it is
 * scanning is coming from the user typing or from a file. However, our input
 * is being provided by Graphviz' I/O channel mechanism, which does not have a
 * valid file descriptor that supports isatty().
 */
#define isatty(x) gv_isatty_suppression
int gv_isatty_suppression;

static int read_input(aagscan_t yyscanner, char *buf, int max_size);

#ifndef YY_INPUT
#define YY_INPUT(buf,result,max_size) \
	if ((result = read_input(yyscanner, buf, max_size)) < 0) \
		YY_FATAL_ERROR( "input in flex scanner failed" )
#endif

/* buffer for arbitrary length strings (longer than BUFSIZ) */

static void beginstr(aagscan_t yyscanner);
static void addstr(aagscan_t yyscanner, char *src);
static void endstr(aagscan_t yyscanner);
static void endstr_html(aagscan_t yyscanner);
static void storeFileName(aagscan_t yyscanner, const char* fname, size_t len);

/* ppDirective:
 * Process a possible preprocessor line directive.
 * aagtext = #.*
 */
static void ppDirective (aagscan_t yyscanner);

/* twoDots:
 * Return true if token has more than one '.';
 * we know the last character is a '.'.
 */
static bool twoDots(aagscan_t yyscanner);

/* chkNum:
 * The regexp for NUMBER allows a terminating letter or '.'.
 * This way we can catch a number immediately followed by a name
 * or something like 123.456.78, and report this to the user.
 */
static int chkNum(aagscan_t yyscanner);

/* The LETTER class below consists of ascii letters, underscore, all non-ascii
 * characters. This allows identifiers to have characters from any
 * character set independent of locale. The downside is that, for certain
 * character sets, non-letter and, in fact, undefined characters will be
 * accepted. This is not likely and, from dot's stand, shouldn't do any
 * harm. (Presumably undefined characters will be ignored in display.) And,
 * it allows a greater wealth of names. */
%}
GRAPH_EOF_TOKEN				[@]	
LETTER [A-Za-z_\200-\377]
DIGIT	[0-9]
NAME	{LETTER}({LETTER}|{DIGIT})*
NUMBER	[-]?(({DIGIT}+(\.{DIGIT}*)?)|(\.{DIGIT}+))(\.|{LETTER})?
ID		({NAME}|{NUMBER})
%x comment
%x qstring
%x hstring
%%
{GRAPH_EOF_TOKEN}		return(EOF);
<INITIAL,comment>\n	yyextra->line_num++;
"/*"					BEGIN(comment);
<comment>[^*\n]*		/* eat anything not a '*' */
<comment>"*"+[^*/\n]*	/* eat up '*'s not followed by '/'s */
<comment>"*"+"/"		BEGIN(INITIAL);
"//".*					/* ignore C++-style comments */
^"#".*					ppDirective (yyscanner);
"#".*					/* ignore shell-like comments */
[ \t\r]					/* ignore whitespace */
"\xEF\xBB\xBF"				/* ignore BOM */
"node"					return(T_node);			/* see tokens in agcanonstr */
"edge"					return(T_edge);
"graph"					if (!yyextra->graphType) yyextra->graphType = T_graph; return(T_graph);
"digraph"				if (!yyextra->graphType) yyextra->graphType = T_digraph; return(T_digraph);
"strict"				return(T_strict);
"subgraph"				return(T_subgraph);
"->"				if (yyextra->graphType == T_digraph) return(T_edgeop); else return('-');
"--"				if (yyextra->graphType == T_graph) return(T_edgeop); else return('-');
{NAME}					{ yylval->str = agstrdup(yyextra->G,aagget_text(yyscanner)); return(T_atom); }
{NUMBER}				{ if (chkNum(yyscanner)) yyless(aagget_leng(yyscanner)-1); yylval->str = agstrdup(yyextra->G,aagget_text(yyscanner)); return(T_atom); }
["]						BEGIN(qstring); beginstr(yyscanner);
<qstring>["]			BEGIN(INITIAL); endstr(yyscanner); return (T_qatom);
<qstring>[\\]["]		addstr (yyscanner,"\"");
<qstring>[\\][\\]		addstr (yyscanner,"\\\\");
<qstring>[\\][\n]		yyextra->line_num++; /* ignore escaped newlines */
<qstring>[\n]			addstr (yyscanner,"\n"); yyextra->line_num++;
<qstring>([^"\\\n]*|[\\])		addstr(yyscanner,aagget_text(yyscanner));
[<]						BEGIN(hstring); yyextra->html_nest = 1; beginstr(yyscanner);
<hstring>[>]			yyextra->html_nest--; if (yyextra->html_nest) addstr(yyscanner,aagget_text(yyscanner)); else {BEGIN(INITIAL); endstr_html(yyscanner); return (T_qatom);}
<hstring>[<]			yyextra->html_nest++; addstr(yyscanner,aagget_text(yyscanner));
<hstring>[\n]			addstr(yyscanner,aagget_text(yyscanner)); yyextra->line_num++; /* add newlines */
<hstring>([^><\n]*)		addstr(yyscanner,aagget_text(yyscanner));
.						return aagget_text(yyscanner)[0];
%%

void aagerror(aagscan_t yyscanner, const char *str)
{
	/* for YYSTATE macro */
	struct yyguts_t *yyg = yyscanner;

	agxbuf xb = {0};
	if (yyextra->InputFile) {
		agxbprint (&xb, "%s: ", yyextra->InputFile);
	}
	agxbprint (&xb, "%s in line %d", str, yyextra->line_num);
	if (*aagget_text(yyscanner)) {
		agxbprint(&xb, " near '%s'", aagget_text(yyscanner));
	}
	else switch (YYSTATE) {
	case qstring: {
		agxbprint(&xb, " scanning a quoted string (missing endquote? longer than %d?)", YY_BUF_SIZE);
		if (agxblen(&yyextra->Sbuf) > 0) {
			agxbprint(&xb, "\nString starting:\"%.80s", agxbuse(&yyextra->Sbuf));
		}
		break;
	}
	case hstring: {
		agxbprint(&xb, " scanning a HTML string (missing '>'? bad nesting? longer than %d?)", YY_BUF_SIZE);
		if (agxblen(&yyextra->Sbuf) > 0) {
			agxbprint(&xb, "\nString starting:<%.80s", agxbuse(&yyextra->Sbuf));
		}
		break;
	}
	case comment :
		agxbprint(&xb, " scanning a /*...*/ comment (missing '*/? longer than %d?)", YY_BUF_SIZE);
		break;
	default: // nothing extra to note
		break;
	}
	agxbputc (&xb, '\n');
	agerrorf("%s", agxbuse(&xb));
	agxbfree(&xb);
    BEGIN(INITIAL);
}
/* must be here to see flex's macro defns */
void aglexeof(aagscan_t yyscanner) {
  struct yyguts_t *yyg = yyscanner;
  unput(GRAPH_EOF_TOKEN);
}

void aglexbad(aagscan_t yyscanner) {
  struct yyguts_t *yyg = yyscanner;
  YY_FLUSH_BUFFER;
}

#ifndef YY_CALL_ONLY_ARG
# define YY_CALL_ONLY_ARG aagscan_t yyscanner
#endif

int aagwrap(YY_CALL_ONLY_ARG)
{
	(void)yyscanner;
	return 1;
}

static void beginstr(aagscan_t yyscanner) {
  aagextra_t *ctx = aagget_extra(yyscanner);
  // avoid unused variable warning in case assert() is compiled away.
  (void)ctx;
  // nothing required, but we should not have pending string data
  assert(agxblen(&ctx->Sbuf) == 0 &&
         "pending string data that was not consumed (missing "
         "endstr()/endhtmlstr()?)");
}

static void addstr(aagscan_t yyscanner, char *src) {
  aagextra_t *ctx = aagget_extra(yyscanner);
  agxbput(&ctx->Sbuf, src);
}

static void endstr(aagscan_t yyscanner) {
  aagextra_t *ctx = aagget_extra(yyscanner);
  aagget_lval(yyscanner)->str = agstrdup(ctx->G, agxbuse(&ctx->Sbuf));
}

static void endstr_html(aagscan_t yyscanner) {
  aagextra_t *ctx = aagget_extra(yyscanner);
  aagget_lval(yyscanner)->str = agstrdup_html(ctx->G, agxbuse(&ctx->Sbuf));
}

static void storeFileName(aagscan_t yyscanner, const char* fname, size_t len) {
	aagextra_t *ctx = aagget_extra(yyscanner);
	agxbuf *buffer = &ctx->InputFileBuffer;
	agxbput_n(buffer, fname, len);
	ctx->InputFile = agxbuse(buffer);
}

/* ppDirective:
 * Process a possible preprocessor line directive.
 * aagtext = #.*
 */
static void ppDirective (aagscan_t yyscanner)
{
    int r, cnt, lineno;
    char buf[2];
    char* s = aagget_text(yyscanner) + 1;  /* skip initial # */

    if (startswith(s, "line")) s += strlen("line");
    r = sscanf(s, "%d %1[\"]%n", &lineno, buf, &cnt);
    if (r > 0) { /* got line number */
	// ignore if line number was out of range
	if (lineno <= 0) {
	    return;
	}
	aagget_extra(yyscanner)->line_num = lineno - 1;
	if (r > 1) { /* saw quote */
	    char* p = s + cnt;
	    char* e = p;
	    while (*e && *e != '"') e++;
	    if (e != p && *e == '"') {
 		*e = '\0';
		storeFileName(yyscanner, p, (size_t)(e - p));
	    }
	}
    }
}

/* twoDots:
 * Return true if token has more than one '.';
 * we know the last character is a '.'.
 */
static bool twoDots(aagscan_t yyscanner) {
  const char *dot = strchr(aagget_text(yyscanner), '.');
  // was there a dot and was it not the last character?
  return dot != NULL && dot != &aagget_text(yyscanner)[aagget_leng(yyscanner) - 1];
}

/* chkNum:
 * The regexp for NUMBER allows a terminating letter or '.'.
 * This way we can catch a number immediately followed by a name
 * or something like 123.456.78, and report this to the user.
 */
static int chkNum(aagscan_t yyscanner) {
    char c = aagget_text(yyscanner)[aagget_leng(yyscanner) - 1]; // last character
    if ((!gv_isdigit(c) && c != '.') || (c == '.' && twoDots(yyscanner))) { // c is letter
	const char* fname;
	aagextra_t *ctx = aagget_extra(yyscanner);

	if (ctx->InputFile)
	    fname = ctx->InputFile;
	else
	    fname = "input";

	agwarningf("syntax ambiguity - badly delimited number '%s' in line %d of "
	  "%s splits into two tokens\n", aagget_text(yyscanner), ctx->line_num, fname);

	return 1;
    }
    else return 0;
}


static int read_input(aagscan_t scanner, char *buf, int max_size)
{
	aagextra_t *ctx = aagget_extra(scanner);
	return ctx->Disc->io->afread(ctx->Ifile, buf, max_size);
}
