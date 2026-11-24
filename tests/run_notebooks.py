"""Execute the Phase 3 notebooks end-to-end and persist their outputs.

Prefers a real Jupyter kernel (nbclient + ipykernel) when available. If the
kernel stack cannot be installed, falls back to executing every code cell, in
order, in one shared namespace per notebook, capturing stdout and embedding any
plot written by the notebooks' ``show()`` helper as a PNG output.

Run: py tests/run_notebooks.py
"""

from __future__ import annotations

import ast
import base64
import io
import json
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    "06_model_training.ipynb",
    "07_model_comparison.ipynb",
    "08_cross_validation.ipynb",
    "09_robustness_testing.ipynb",
    "10_error_analysis.ipynb",
]


def _kernel_available() -> bool:
    try:
        import nbclient  # noqa: F401
        import nbformat  # noqa: F401
        import ipykernel  # noqa: F401
    except Exception:
        return False
    return True


def execute_with_kernel(path: Path) -> None:
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook, timeout=900, kernel_name="python3", allow_errors=False,
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    nbformat.write(notebook, path)


def _run_source(source: str, namespace: dict, filename: str) -> None:
    """Exec a cell; print the value of a trailing expression like a kernel does."""
    tree = ast.parse(source, filename=filename)
    trailing = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        trailing = tree.body.pop()
    exec(compile(tree, filename, "exec"), namespace)
    if trailing is not None:
        value = eval(compile(ast.Expression(trailing.value), filename, "eval"), namespace)
        if value is not None:
            print(value if isinstance(value, str) else repr(value))


def _stream(text: str) -> dict:
    return {"output_type": "stream", "name": "stdout", "text": text.splitlines(keepends=True)}


def _build_outputs(text: str) -> list[dict]:
    outputs: list[dict] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            outputs.append(_stream("".join(buffer)))
            buffer.clear()

    for line in text.splitlines(keepends=True):
        if line.startswith("PLOT:"):
            flush()
            target = Path(line[len("PLOT:"):].strip())
            try:
                encoded = base64.b64encode(target.read_bytes()).decode("ascii")
            except OSError:
                buffer.append(f"missing plot: {target}\n")
                continue
            outputs.append(
                {
                    "output_type": "display_data",
                    "data": {"image/png": encoded, "text/plain": ["<figure>"]},
                    "metadata": {},
                }
            )
        else:
            buffer.append(line)
    flush()
    return outputs


def execute_inline(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict = {"__name__": "__main__", "__file__": str(path)}
    counter = 0
    failed = False

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        counter += 1
        cell["execution_count"] = counter
        cell["outputs"] = []
        if failed:
            cell["execution_count"] = None
            continue
        source = "".join(cell["source"])
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                _run_source(source, namespace, str(path))
        except Exception:
            detail = traceback.format_exc()
            cell["outputs"] = [
                {
                    "output_type": "error",
                    "ename": "NotebookCellExecutionError",
                    "evalue": detail.strip().splitlines()[-1],
                    "traceback": detail.strip().splitlines(),
                }
            ]
            failed = True
            continue
        captured = buffer.getvalue()
        if captured:
            cell["outputs"] = _build_outputs(captured)

    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    if failed:
        raise RuntimeError(f"cell {counter} failed in {path.name}")


def main() -> int:
    mode = "jupyter kernel (nbclient)" if _kernel_available() else "builtin executor"
    print(f"Notebook execution mode: {mode}")
    failures = []
    for name in NOTEBOOKS:
        path = ROOT / "notebooks" / name
        if not path.exists():
            failures.append((name, "missing"))
            print(f"MISSING  {name}")
            continue
        try:
            if _kernel_available():
                execute_with_kernel(path)
            else:
                execute_inline(path)
            notebook = json.loads(path.read_text(encoding="utf-8"))
            code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
            errors = [o for c in code_cells for o in c["outputs"] if o.get("output_type") == "error"]
            if errors:
                failures.append((name, errors[0].get("evalue", "error")))
                print(f"FAIL  {name}: {errors[0].get('evalue')}")
            else:
                outputs = sum(len(c["outputs"]) for c in code_cells)
                print(f"PASS  {name}: {len(code_cells)} code cells executed, {outputs} outputs")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, str(exc)))
            print(f"FAIL  {name}: {exc}")

    print("-" * 60)
    print(f"{len(NOTEBOOKS) - len(failures)}/{len(NOTEBOOKS)} notebooks executed cleanly")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
