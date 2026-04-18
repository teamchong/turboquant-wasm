/**
 * CodeMirror 6 editor for generated diagram code.
 * Editable — user can modify code, changes sync to currentCode.
 * Model sees manual edits when generating next iteration.
 */

import { EditorView, keymap } from "@codemirror/view";
import { EditorState } from "@codemirror/state";
import { javascript } from "@codemirror/lang-javascript";
import { oneDark } from "@codemirror/theme-one-dark";
import { defaultKeymap, history, historyKeymap } from "@codemirror/commands";
import { syntaxHighlighting, defaultHighlightStyle } from "@codemirror/language";
import { lineNumbers } from "@codemirror/view";

let view: EditorView | null = null;
let onChange: ((code: string) => void) | null = null;

export function createEditor(container: HTMLElement, onCodeChange: (code: string) => void): EditorView {
  onChange = onCodeChange;

  const state = EditorState.create({
    doc: "",
    extensions: [
      lineNumbers(),
      history(),
      javascript(),
      oneDark,
      syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
      keymap.of([...defaultKeymap, ...historyKeymap]),
      EditorView.updateListener.of((update) => {
        if (update.docChanged && onChange) {
          onChange(update.state.doc.toString());
        }
      }),
      EditorView.theme({
        "&": { height: "100%", fontSize: "12px" },
        ".cm-scroller": { overflow: "auto" },
      }),
    ],
  });

  view = new EditorView({ state, parent: container });
  return view;
}

export function setCode(code: string) {
  if (!view) return;
  view.dispatch({
    changes: { from: 0, to: view.state.doc.length, insert: code },
  });
}

export function getCode(): string {
  if (!view) return "";
  return view.state.doc.toString();
}

export function appendCode(chunk: string) {
  if (!view) return;
  view.dispatch({
    changes: { from: view.state.doc.length, insert: chunk },
  });
}
