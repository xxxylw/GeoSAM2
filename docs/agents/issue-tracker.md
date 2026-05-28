# Issue tracker: GitHub

Issues and PRDs for this repo live as GitHub issues by default. Use the `gh` CLI for issue operations when credentials and repository permissions are available.

## Conventions

- Create an issue with `gh issue create --title "..." --body-file <file>`.
- Read an issue with `gh issue view <number> --comments`.
- List issues with `gh issue list` and appropriate labels or state filters.
- Comment with `gh issue comment <number> --body "..."`.
- Apply labels with `gh issue edit <number> --add-label "..."`.

When a skill says "publish to the issue tracker", create a GitHub issue. If GitHub credentials or write permissions are unavailable, write the PRD or issue body under `docs/prd/` and report that it was not published remotely.
