# Spec: <Feature Name>

**ID:** `NNN` _(increment from last spec in `docs/specs/`)_
**Status:** `Draft` | `In Progress` | `Implemented`
**Author:** _(optional)_
**Created:** YYYY-MM-DD
**Last updated:** YYYY-MM-DD

---

## Purpose

> What problem does this solve? Why is this being built now?
> One short paragraph. Be specific about the gap, not just the solution.

---

## Affected Modules

> List every directory, file, or module that this change touches or depends on.

- `module_a/` — reason
- `module_b/schemas/` — reason
- `utils/llm_clients.py` — reason

---

## Out of Scope

> What is explicitly NOT being changed in this task.
> This prevents scope creep and tells the agent what not to touch.

- Not changing X.
- Not refactoring Y.

---

## Implementation Plan

> Ordered step-by-step changes. Be specific enough that an agent can follow this.

1. Step one.
2. Step two.
3. Step three.

---

## Schemas / Data Contracts

> New or modified dataclasses, Pydantic models, config structures, prompt formats, or output formats.
> If unchanged, write "No schema changes."

```python
# Example new schema
class FeatureOutput(BaseModel):
    field_a: str
    field_b: int
    metadata: dict[str, Any]
```

---

## Validation

> How will you prove this works after implementation?

- **Command:** `python -m module_name.runners run_feature --input test_input.json`
- **Notebook:** `notebooks/feature_name_inspection.ipynb` — check section 3 output matches expected shape.
- **Test:** `pytest tests/test_feature_name.py`
- **Expected output:** Describe what correct output looks like.

---

## Known Risks / Limitations

> Failure modes, edge cases, or performance concerns to be aware of.

- Risk 1: ...
- Risk 2: ...

---

## Open Questions

> Unresolved decisions or things that need human input before/during implementation.

- [ ] Question 1
- [ ] Question 2

---

## Implementation Notes

> Filled in during or after implementation. What diverged from the plan? Why?
> Leave blank until implementation begins.
