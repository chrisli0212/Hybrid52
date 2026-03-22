# Agent / AI coordination (Hybrid52)


## Git — do **not** lock the repo for other sessions


This repository may be updated from **multiple environments** (e.g. RunPod + GitHub + local). Follow these rules so another assistant (or human) can **pull, merge, and push** without fighting a half-finished Git state.


### Required


1. **Never leave Git in a stuck state**  
   If you start `git rebase`, `git merge`, or resolve conflicts, **finish or abort** before ending your turn:
   - `git rebase --continue` / `git merge --continue` when done, or  
   - `git rebase --abort` / `git merge --abort` if you are not completing it.


2. **Do not "lock" by long operations**  
   Avoid leaving an incomplete interactive rebase, unresolved conflict markers (`<<<<<<<`), or a dirty index that only your session knows how to fix.


3. **Prefer integration over force**  
   - Use `git pull` (merge or rebase) and resolve conflicts **in the working tree**, then commit.  
   - **Do not** `git push --force` or rewrite `main` history unless the **human owner** explicitly asks.


4. **Credentials**  
   Do **not** store GitHub tokens in `git remote` URLs. Use SSH, `gh auth`, or environment/credential helpers. Rotate any token that was ever pasted into a URL.


5. **Source of truth**  
   The **RunPod workspace** may be ahead of GitHub; treat GitHub as a mirror until changes are pushed. Another session may need to push — keep `main` mergeable.


### If you must pause mid-merge


- Abort (`--abort`) **or** commit the merge with a clear message.  
- Optionally add a one-line note in the PR/commit body: *"WIP merge — follow-up push expected."*


---


*This file is for humans and AI assistants. It is not enforced by tooling; it prevents accidental repo-wide locks.*
