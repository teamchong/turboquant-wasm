#!/usr/bin/env python3

"""entities.h generator"""

import re
import sys
import textwrap
from pathlib import Path

# get names for html-4.0 characters from:
#          http://www.w3.org/TR/REC-html40/sgml/entities.html
entity_name_length_max: int = 0
entity: dict[str, str] = {}
with open(Path(__file__).parent / "entities.html", "rb") as f:
    for rec in f:
        if m := re.match(
            rb'&lt;!ENTITY\s+(?P<name>[^\s]*)\s+CDATA\s+"&amp;#(?P<val>\d+);"\s+--', rec
        ):
            name = m.group("name").decode("utf-8")
            val = m.group("val").decode("utf-8")
            entity[name] = val
            entity_name_length_max = max(entity_name_length_max, len(name))

with open(sys.argv[1], "wt", encoding="utf-8") as f:
    f.write(
        textwrap.dedent(
            f"""\
    /// @file
    /// @ingroup common_utils
    /*
     * Generated file - do not edit directly.
     *
     * This file was generated from:
     *       http://www.w3.org/TR/REC-html40/sgml/entities.html
     * by means of the script:
     *       {Path(__file__).name}
     */

    #ifdef __cplusplus
    extern "C" {{
    #endif

    static const struct entities_s {{
    	char	*name;
    	int	value;
    }} entities[] = {{
    """
        )
    )
    for name, val in sorted(list(entity.items())):
        f.write(f'	{{"{name}", {val}}},\n')
    f.write(
        textwrap.dedent(
            f"""\
    }};

    #define ENTITY_NAME_LENGTH_MAX {entity_name_length_max}
    #define NR_OF_ENTITIES (sizeof(entities) / sizeof(entities[0]))

    #ifdef __cplusplus
    }}
    #endif
    """
        )
    )
