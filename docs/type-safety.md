# Type Safety

This document describes the pipeline's approach to **static typing**: what gets
annotated, how the intentionally dynamic boundaries are handled, and how the type
checker is configured. It is the reference for contributors adding typed code and
the companion to the [Exception Strategy](exception-strategy.md).

---

## 1. Goals

- **Annotate every public interface.** Each module-level function carries
  parameter and return annotations, so callers and IDEs know the contract
  without reading the body.
- **Fail at the type checker, not at runtime.** A strict [mypy](https://mypy.readthedocs.io/)
  configuration ([`pyproject.toml`](../pyproject.toml)) catches type mismatches
  before the pipeline runs.
- **No unnecessary `Any`.** `Any` is used only where a value is genuinely dynamic
  (a deserialized artifact, a YAML value); everywhere else the concrete type — or
  `object` — is used instead.
- **No suppressions.** Type errors are fixed by improving the types, never by
  adding `# type: ignore` or `cast(...)` workarounds.
- **Document the hard cases.** Where inference is impossible, a docstring or
  comment explains *why* the type is what it is.

---

## 2. Conventions

- **Modern generics.** Abstract collection types come from
  `collections.abc` (`Sequence`, `Callable`), not their deprecated `typing`
  aliases. Built-in generics are subscripted directly (`dict[str, Any]`,
  `list[int]`).
- **PEP 604 unions.** Optionals are written `int | None`, not
  `Optional[int]` (e.g. `max_depth: int | None` in
  [`train.py`](../src/train.py)).
- **`object` over `Any` for opaque inputs.** A function that accepts a value but
  makes no assumptions about it takes `object`, which still accepts anything but
  forbids the caller-invisible operations that `Any` would silently allow (see
  `save_pickle` in [`pipeline_io.py`](../src/pipeline_io.py)).

---

## 3. The dynamic boundaries — where `Any` is deliberate

Three boundaries are dynamic by nature. At each, the wider type is a documented,
deliberate choice — not missing rigor.

| Location | Type | Why it is not narrower |
|----------|------|------------------------|
| `load_params(...) -> dict[str, Any]` | `Any` values | YAML is schemaless; a parameter's concrete type is known only to the stage that consumes it, where it flows into a typed function call and is checked there. |
| `load_pickle(...) -> Any` | `Any` return | The on-disk type is not known statically; callers use the result as the concrete artifact they expect (e.g. an estimator with `.predict`). |

These are the only two `Any`s in the pipeline's own code.

**Narrowing vs. inference.** The `param_grid` in [`train.py`](../src/train.py)
looks heterogeneous but every candidate is an `int` or `None`, so it is typed
`dict[str, list[int | None]]` — not `list[Any]`. It still needs an *explicit*
annotation on the literal: without one, its value lists (several `list[int]` plus
one `list[int | None]`) infer as the invariant join `dict[str, object]`, which
would not match the helper's signature — a real error mypy flags. Annotating the
literal both narrows the type and resolves the inference.

---

## 4. Type checking

Configuration lives in [`pyproject.toml`](../pyproject.toml) under
`[tool.mypy]`. Run it from the repository root:

```bash
pip install -r requirements-dev.txt   # installs mypy
python -m mypy                         # checks src/ per pyproject.toml
```

Key settings and their rationale:

| Setting | Effect |
|---------|--------|
| `disallow_untyped_defs`, `disallow_incomplete_defs` | Every function must be fully annotated. |
| `warn_return_any` | Flags a function that silently returns an `Any` where a concrete type was declared. |
| `no_implicit_optional`, `strict_equality`, `warn_unreachable` | Catch a class of subtle logic/type bugs. |
| `warn_unused_ignores` | Ensures suppressions (which this project avoids) cannot rot. |
| `mypy_path = "src"`, `explicit_package_bases` | Resolves the stages' bare sibling imports (`from exceptions import ...`) the same way the interpreter does when a stage runs as a script. |

### The third-party boundary

The ML and tracking dependencies are not type-checked here, for one of two
reasons:

- **No type information at all** — `mlflow` and `scikit-learn` ship neither a
  `py.typed` marker nor a published stub package.
- **Stubs available but not installed** — `pandas` and `yaml` have maintained
  stub packages (`pandas-stubs`, `types-PyYAML`) that are not currently
  dependencies.

(`dvc` is used only as a command-line tool here, never imported, so it is not
part of the type-checked surface at all.)

All are listed under a `[[tool.mypy.overrides]]` block with
`ignore_missing_imports = true`, which scopes "untyped" strictly to those
libraries' own APIs. It does **not** weaken checking of the pipeline's code: our
functions remain fully annotated, and values crossing that boundary are given
explicit types (as in §3). Adding `pandas-stubs` and `types-PyYAML` to
type-check DataFrame and YAML usage directly is a possible future enhancement.
(`python-dotenv`, by contrast, ships its own types and *is* checked.)

---

## 5. Adding new code — checklist

1. **Annotate the signature** — every parameter and the return type.
2. **Use the narrowest honest type** — a concrete type where you know it,
   `object` where you accept anything but assume nothing, `Any` only at a truly
   dynamic boundary.
3. **Prefer `collections.abc` and `|` unions** over the `typing` aliases and
   `Optional`.
4. **Document a wide type.** If you must use `Any`/`object`, say why in the
   docstring or a comment.
5. **Run `python -m mypy`** and fix findings by improving types — never by
   suppressing them.

---

## Related Documentation

- [Exception Strategy](exception-strategy.md) — the typed error hierarchy.
- [Logging Strategy](logging.md) — format, levels, and destinations.
- [Architecture](architecture.md) — system overview.
