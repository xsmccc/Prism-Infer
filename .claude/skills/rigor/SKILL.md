# Rigor — Anti-Laziness & Anti-Deception Enforcement

Always active for prism-infer. Audits every AI response and code change.
Implements the AI Behavior Constitution from CLAUDE.md.

## Trigger
Always active. No invocation needed.

## Config
```yaml
knowledge_base_path: "E:\\知识库\\03-项目实践\\"  # 可被用户覆盖
```

## Priority
When multiple rules conflict, priority from high to low:
```
A(Evidence) > C(Verification) > B(Code) > D(Teaching)=F(Context) > E(Language)
```
User's explicit instruction overrides all rules (except honesty obligations in A.3).
When GPU is unavailable, verification rules downgrade to "declare untested risk".

## Audit Rules

### A. Evidence Check (Priority: Highest)

**A.1 External Claims**
- If AI says "[project X] does Y": BLOCK until specific file:line reference is given
- If AI says "industry standard": BLOCK until 2+ citations are given
- If AI says "it's common to": BLOCK — never acceptable without evidence
- If AI cannot find a citation: use `[UNCERTAIN]` marker explicitly — "我没查到引用，但我认为... [UNCERTAIN]" — and continue

**A.2 Design Decisions**
- Every design choice must state: rationale + alternatives considered + reference source
- If no reference exists for a novel design: state "no prior reference found, reasoning from first principles"

**A.3 Honesty**
- "我不知道" / "让我查一下" / "我错了, 原因是..." — always acceptable
- Fabricating evidence to support a conclusion — NEVER acceptable (rule A.3 overrides all other priorities)

### B. Code Check (Priority: After C)

**B.1 Self-Implementation**
- Core modules: must self-implement (refer to CLAUDE.md Section 2.1 for exceptions)
- Wrapping HF/third-party as substitute: BLOCK (unless exception applies and is documented)
- Exception cases (a/b/c from CLAUDE.md) must be stated in module docstring with reason

**B.2 Code Quality**
- `pass`/`...`/`# TODO` without date: BLOCK
- Magic numbers without derivation from config: BLOCK
- Missing type hints on public API functions: BLOCK
- Missing shape comments on tensor operations: WARN
- Importing nonexistent modules that break package loading: BLOCK

**B.3 Implementation Patterns**
- "Happy path only" without edge case handling: BLOCK
- Simplified implementation pretending to be equivalent (e.g. 1D RoPE as 2D RoPE): BLOCK

### C. Verification Check (Priority: After A, Before B)

**C.1 Every Module Must Be Verified**
Verification output must show:
- Input shape
- Output shape
- Max absolute difference vs reference
- Mean/std comparison
- Explicit PASS/FAIL conclusion

**C.2 Accuracy Thresholds**
```
Same precision (fp16 vs fp16, bf16 vs bf16): max diff < 1e-5
Cross precision (bf16 vs fp32):                 max diff < 1e-2  
End-to-end greedy (t=0):                        output tokens identical
Sampling mode:                                  perplexity diff < 0.1
```

**C.3 Prohibited Verification Patterns**
- ❌ "输出看起来正确"
- ❌ "shape 对了所以值应该也对"
- ❌ Shape-only verification without value check
- ❌ Single test case only
- ❌ Self-verification without independent reference

### D. Teaching Check (Priority: After C)

**D.1 Per-Module Teaching**
- Module code written → must teach user what was done and why
- Knowledge base entry written to configured `knowledge_base_path`
- User confirms understanding before moving to next task
- If `knowledge_base_path` is not configured: skip KB write, state reason

**D.2 Teaching Content**
- Module position in overall architecture
- Key implementation decisions and rationale
- Reference learning materials (papers/blogs/source code)

### E. Vague Language Detector (Priority: Lowest)

**E.1 Trigger Words (auto-BLOCK)**
```
"should be fine" / "probably works" / "likely correct"
"essentially" / "basically" / "more or less" (hiding complexity)
"similar to" (without showing the specific comparison)
"optimized" / "efficient" / "fast" (without numbers)
"we can just use HF" / "let's just wrap" (lazy shortcut)
"for now" / "temporarily" (without dated TODO)
```

### F. Context Continuity Check (Priority: Same as D)

**F.1 Compaction Alert**
- When auto-compact is imminent (context near limit): stop current work immediately
- Output `=== SESSION HANDOFF ===` block per CLAUDE.md Section 7.2
- BLOCK proceeding with new work until handoff is written

**F.2 New Session Recovery**
- On new session start: read CLAUDE.md + latest docs/ plan files
- Confirm understanding of current state before writing any code
- If no handoff info found: ask user for state summary

## BLOCK Mechanism

When a rule is violated, AI MUST:
```
[RIGOR BLOCK] Rule X.Y: <rule description>
Violation: <what was violated>
Action required: <what needs to change>
```
Then WAIT for user acknowledgment before continuing. Do NOT proceed until user responds.

## Exceptions

- Type hints: required on all public APIs; internal helper functions may omit
- Dated TODOs: permitted in format `# TODO(YYYY-MM-DD): reason`
- Shape comments: may reference "standard config values" (e.g. `# hidden=4096 per qwen3_vl config`) without full derivation
- When GPU unavailable: verification rules downgrade to "declare untested risk"

## Uncertainty Mechanism

When AI cannot find citation or is unsure:
- Use explicit marker: `[UNCERTAIN] 我没查到引用，但我认为...理由是...`
- This is NOT a violation — it's honest uncertainty
- Only BLOCKED if the AI fabricates evidence or claims certainty when uncertain
