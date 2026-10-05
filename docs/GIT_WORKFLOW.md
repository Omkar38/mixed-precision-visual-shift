# Git identity and publication

Omkar requested his identity, rather than Codex or Claude, for commits made on his behalf.

## Identity

Use repository-local configuration:

```bash
git config --local user.name "Omkar Lashkare"
git config --local user.email "omkarvilas.lashkare@stonybrook.edu"
git var GIT_AUTHOR_IDENT
git var GIT_COMMITTER_IDENT
```

This email is verified against the project proposal, not against GitHub. For GitHub to associate commits with Omkar's profile, it must be one of his verified account emails. If Omkar provides his GitHub no-reply address instead, use that exact address for future commits; do not invent one.

Environment variables such as GIT_AUTHOR_NAME or GIT_COMMITTER_EMAIL can override local configuration. Inspect effective identity before committing; do not print credentials. The initial commit explicitly sets both author and committer to Omkar. Do not append a Codex/Claude co-author trailer.

This convention applies to work performed on Omkar's behalf. Preserve other contributors' authorship. Do not globally change their identity or rewrite shared history.

## Publish under the actual authenticated account

Git commit identity and the account that pushes are separate. Never assume a GitHub username from the name or email. Verify the authenticated account first.

Once GitHub CLI and account access are available, these commands create a **private** repository from this directory. They are instructions, not evidence that publication has occurred:

```bash
gh auth status
gh api user --jq .login
gh repo create mixed-precision-visual-shift --private --source=. --remote=origin --push
git remote -v
git log -1 --format=fuller
```

Verify the displayed login is Omkar's intended account before creation. If a repository or origin already exists, inspect and use it rather than overwriting it. Do not make the repository public or add collaborators without direction.

## Multiple agents

Use separate branches/worktrees for concurrent sessions. Do not run Claude Code and Codex against the same writable checkout at the same time. Integrate small commits after checking tests and scientific assumptions.

## Licensing

No project code license has been selected. Do not add a permissive license on behalf of the whole team without agreement. Dataset and model licenses remain separate.
