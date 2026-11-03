# Rule: Prohibit Remote Git Push

1. **Strict Prohibition on `git push`:**
   - Under NO circumstances should the AI agent execute `git push` (e.g., `git push origin master`, `git push origin <branch>`, `git push --tags`, etc.).
   - Pushing code to any remote repository is exclusively reserved for the USER following their manual review of changes.

2. **Local Operations Only:**
   - The agent may inspect git status, create local branches, make local file edits, run tests, and stage/commit locally if requested, but MUST stop before pushing.
   - When changes are ready or committed locally, notify the user so they can review and push manually.
