# Session instructions

- Read `REPO_MAP.md` at session start before exploring the codebase. Use it as the
  map; only grep/read specific files when you need details beyond what it says.
- Update `REPO_MAP.md` when files are added, removed, or their purpose changes.
  It describes only the **current state** of the repo: no dated change history,
  no commit references, no "was previously / legacy" comparisons, and **never
  plan files** (plans like `PLAN_*.md` are temporary and deleted after
  implementation — they must not appear in the map).
- Commit to git after every coherent unit of work. Stage only the intended
  files, verify `git status --short` is clean after, and never leave the tree
  dirty at the end of a task. Never stage any gitignored generated artifact
  (downloaded data, built databases, image caches, job outputs).
