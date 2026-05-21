"""
Extract all benchmark prompt templates from llm_eval/benchmarks.

Walks the benchmarks package and finds module-level assignments to identifiers
that look like prompt templates (prompt_template, PROMPT_TEMPLATE(S),
JUDGE_PROMPT, ...). Statically evaluates the right-hand side via ast.literal_eval
when possible so we don't need to import or run any benchmark code.

Some benchmarks (BZK social bias, Dutch BBQ, Dutch CrowS-Pairs, HonestCity)
do not use a template string -- the prompt is the raw `question` / `prompt`
field of each dataset row. These are reported as "data-driven" so they show
up in the output too.

Usage:
    python scripts/extract_prompt_templates.py                # text to stdout
    python scripts/extract_prompt_templates.py --format latex # LaTeX to stdout
    python scripts/extract_prompt_templates.py -o prompts.tex --format latex
"""
import argparse
import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = REPO_ROOT / "llm_eval" / "benchmarks"

# Identifiers we treat as prompt templates.
TEMPLATE_NAME_RE = re.compile(
    r"^(prompt_template|PROMPT_TEMPLATE|PROMPT_TEMPLATES|"
    r"JUDGE_PROMPT|EVAL(UATION)?_PROMPT|TEMPLATE)s?$"
)

# Files where the prompt comes from the dataset rather than a literal template.
# We list them explicitly so they still appear in the report.
DATA_DRIVEN_NOTES: Dict[str, str] = {
    "social_bias/bzk_social_bias.py": (
        "Prompts read from CSV column `prompt` (renateburema/master_thesis). "
        "No literal template in code."
    ),
    "social_bias/dutch_bbq.py": (
        "Prompts read from dataset field `question_full`. No literal template in code."
    ),
    "social_bias/dutch_crowspairs.py": (
        "Prompts read from dataset field `question`. No literal template in code."
    ),
    "honesty/honest_city_bench.py": (
        "Prompts read from dataset field `prompt_cleaned`. No literal template in code."
    ),
}


@dataclass
class TemplateEntry:
    """A single prompt template found in a benchmark file."""

    name: str
    value: Union[str, Dict, List, None]  # literal text, or nested dict/list of texts
    note: Optional[str] = None  # for data-driven entries


@dataclass
class BenchmarkFile:
    """All template entries discovered in one .py file."""

    rel_path: str
    templates: List[TemplateEntry] = field(default_factory=list)


def _try_literal_eval(node: ast.AST) -> Optional[Union[str, Dict, List]]:
    """
    Evaluate `node` to a Python literal. Supports plain strings,
    parenthesised string concatenations (Python folds these at parse time
    into a single Constant), dicts/lists of strings, and f-strings whose
    parts are all constant (rare, but we accept them by joining).
    """
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        pass

    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            elif isinstance(v, ast.FormattedValue):
                # Keep the placeholder visible so the template stays readable.
                src = ast.unparse(v.value) if hasattr(ast, "unparse") else "?"
                parts.append("{" + src + "}")
            else:
                return None
        return "".join(parts)

    return None


# Matches `# "..."` or `# '...'` on its own line (template fragments that were
# commented out, e.g. the English variants in arc.py / mmlu.py).
COMMENTED_STRING_RE = re.compile(r'^\s*#\s*(".*?"|\'.*?\')\s*,?\s*$')


def _extract_commented_strings(source: str, start: int, end: int) -> Optional[str]:
    """Concatenate any `# "..."` lines inside the given 1-indexed line range."""
    lines = source.splitlines()[start - 1 : end]
    parts = []
    for line in lines:
        m = COMMENTED_STRING_RE.match(line)
        if not m:
            continue
        try:
            parts.append(ast.literal_eval(m.group(1)))
        except (ValueError, SyntaxError):
            continue
    return "".join(parts) if parts else None


def extract_from_file(path: Path) -> List[TemplateEntry]:
    """Return all module-level template assignments in `path`."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    found: List[TemplateEntry] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if not TEMPLATE_NAME_RE.match(target.id):
                continue
            value = _try_literal_eval(node.value)
            if value is None:
                continue
            found.append(TemplateEntry(name=target.id, value=value))

            # Also pick up any commented-out fragments inside the assignment --
            # arc.py and mmlu.py keep their English versions this way.
            end_line = node.end_lineno or node.lineno
            commented = _extract_commented_strings(source, node.lineno, end_line)
            if commented:
                found.append(
                    TemplateEntry(name=f"{target.id} (commented-out variant)", value=commented)
                )
    return found


def collect_all() -> List[BenchmarkFile]:
    """Walk the benchmarks directory and collect template entries per file."""
    results: List[BenchmarkFile] = []
    for path in sorted(BENCHMARKS_DIR.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        rel = path.relative_to(BENCHMARKS_DIR).as_posix()
        entries = extract_from_file(path)

        note = DATA_DRIVEN_NOTES.get(rel)
        if not entries and note is None:
            continue  # no templates and not flagged as data-driven -- skip

        bf = BenchmarkFile(rel_path=rel, templates=entries)
        if note is not None:
            bf.templates.append(TemplateEntry(name="(data-driven)", value=None, note=note))
        results.append(bf)
    return results


# ---------- formatters ----------

def _flatten(value, prefix=""):
    """Yield (label, text) tuples for str values inside arbitrarily-nested dicts/lists."""
    if isinstance(value, str):
        yield prefix, value
    elif isinstance(value, dict):
        for k, v in value.items():
            sub = f"{prefix}.{k}" if prefix else str(k)
            yield from _flatten(v, sub)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            sub = f"{prefix}[{i}]" if prefix else f"[{i}]"
            yield from _flatten(v, sub)


def format_text(results: List[BenchmarkFile]) -> str:
    out = []
    for bf in results:
        out.append("=" * 80)
        out.append(f"FILE: llm_eval/benchmarks/{bf.rel_path}")
        out.append("=" * 80)
        for entry in bf.templates:
            if entry.note:
                out.append(f"\n[{entry.name}] {entry.note}\n")
                continue
            for label, text in _flatten(entry.value):
                header = entry.name if not label else f"{entry.name} :: {label}"
                out.append(f"\n--- {header} ---")
                out.append(text)
            out.append("")
    return "\n".join(out)


def _latex_escape(s: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(c, c) for c in s)


def format_latex(results: List[BenchmarkFile]) -> str:
    """LaTeX output using `lstlisting` so prompt text is verbatim."""
    out = [
        "% Auto-generated by scripts/extract_prompt_templates.py",
        "% Requires: \\usepackage{listings}",
        "\\lstset{",
        "  basicstyle=\\ttfamily\\small,",
        "  breaklines=true,",
        "  breakatwhitespace=false,",
        "  columns=fullflexible,",
        "  frame=single,",
        "  showstringspaces=false,",
        "}",
        "",
    ]
    for bf in results:
        section = _latex_escape(bf.rel_path)
        out.append(f"\\section*{{{section}}}")
        for entry in bf.templates:
            if entry.note:
                out.append(
                    f"\\paragraph{{{_latex_escape(entry.name)}}} "
                    f"{_latex_escape(entry.note)}"
                )
                out.append("")
                continue
            for label, text in _flatten(entry.value):
                header = entry.name if not label else f"{entry.name} :: {label}"
                out.append(f"\\subsection*{{{_latex_escape(header)}}}")
                out.append("\\begin{lstlisting}")
                out.append(text)
                out.append("\\end{lstlisting}")
                out.append("")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format", choices=["text", "latex"], default="text",
        help="output format (default: text)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="write to file instead of stdout",
    )
    args = parser.parse_args()

    results = collect_all()
    rendered = format_text(results) if args.format == "text" else format_latex(results)

    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote {args.output} ({len(results)} files)", file=sys.stderr)
    else:
        sys.stdout.write(rendered)


if __name__ == "__main__":
    main()