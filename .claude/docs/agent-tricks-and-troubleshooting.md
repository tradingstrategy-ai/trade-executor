# Agent tricks and troubleshooting

This note covers practical ways to use CLI engineering agents. Read it before
invoking Kimi Code non-interactively from a shell script or another agent.

## Review access and execution policy

Apply this policy to every CLI review agent, including Kimi. A reviewer may use
local read-only inspection, the web, GitHub, and installed skills when they
help verify a finding. GitHub access must be read-only: inspect repositories,
pull requests, issues, workflows, and logs, but never create, comment, label,
merge, push, or otherwise change GitHub state. Keep credentials and raw
provider output private.

The review scope is existing code and existing results only. Reviewers must
never initiate test runs or other long-running operations: do not run test
suites, benchmarks, builds, linters, type checkers, deployments, migrations,
data downloads, model training, development servers, or application commands.
They may inspect already-produced diffs, logs, reports, artifacts, and CI
results. A reviewer that needs evidence outside that scope must describe the
gap instead of starting work to fill it.

Use this instruction in every review prompt:

```text
You are a read-only reviewer. You may inspect local code and existing results,
use web and read-only GitHub access for relevant verification, and use installed
skills when useful. Do not read secrets or credentials. Do not edit, create,
delete, upload, install, authenticate, or otherwise mutate local files, Git
state, GitHub state, credentials, or external systems. Review only existing
code and results: never start tests, benchmarks, builds, linters, type checks,
deployments, migrations, downloads, training, servers, or application commands.
Report any missing evidence instead of running work to obtain it.
```

This is a behavioural instruction, not an OS-level sandbox. The caller must
still select read-only tool permissions or a read-only sandbox when its CLI
supports one.

## Kimi Code CLI

`kimi` is the Kimi Code terminal agent. Verify the binary and configuration
without exposing credentials:

```shell
command -v kimi
kimi --version
kimi doctor
kimi provider list
```

If `kimi` is not on `PATH`, a user-scope installation may still be available
at `~/.kimi-code/bin/kimi`:

```shell
~/.kimi-code/bin/kimi --version
```

If that works, add `~/.kimi-code/bin` to the shell's `PATH` or invoke the
binary by its full path.

These are setup-time checks for the person or automation launching a reviewer;
they are not reviewer actions. A reviewer must continue to follow the shared
review policy above.

Sign in when required:

```shell
kimi login
```

The persistent user configuration lives under `~/.kimi-code/`; do not paste
`config.toml`, OAuth files, or raw `kimi provider list --json` output into a
PR or chat because provider output can contain credential fields.

Non-interactive execution and structured output:

```shell
kimi -p "Say OK"
kimi -p "$(cat /tmp/review-prompt.md)" --output-format stream-json
KIMI_MODEL_THINKING_EFFORT=high \
  kimi --model kimi-code/k3 -p "Review this plan" --output-format stream-json
```

### Thinking effort

The current [Kimi Code model table](https://www.kimi.com/code/docs/en/) lists
`low`, `high`, and `max` thinking effort for `k3`, `k3-256k`, and
`kimi-for-coding`. It does not list a thinking-effort setting for
`kimi-for-coding-highspeed`; leave that model at its provider default unless
the configured provider documents a supported override.

Use `high` by default in Kimi invocation instructions. It is the project
default for normal reviews: use `low` only when turnaround is more important
than depth, and `max` only when the additional latency and quota cost are
justified. Set the level for a single invocation with
`KIMI_MODEL_THINKING_EFFORT`:

```shell
KIMI_MODEL_THINKING_EFFORT=high \
  kimi --model kimi-code/k3 -p "Review the current worktree"
```

The [environment-variable reference](https://www.kimi.com/code/docs/en/kimi-code-cli/configuration/env-vars.html)
accepts a broader generic vocabulary for `KIMI_MODEL_THINKING_EFFORT`, but do
not request an effort level that the selected model does not advertise.

Important Kimi-specific behavior:

- `-p` takes the prompt as an argument. To load a prompt from a file, pass its
  contents as one safely quoted argument, as in `-p "$(cat prompt.md)"`.
- `--output-format stream-json` writes JSONL to stdout. It can contain
  assistant content records, assistant tool-call records, and tool-result
  records. When presenting a review result, extract only final assistant
  content and treat every other record as non-user output. The
  [command reference](https://www.kimi.com/code/docs/en/kimi-code-cli/reference/kimi-command.html)
  documents the stream format and stderr progress behavior.
- Assistant content may not appear immediately in a live `stream-json` run
  because hidden thinking is not emitted. Treat a live process that has emitted
  only non-assistant output as still running, not as an auth or parser failure.
- The official model IDs are documented as `k3`, `k3-256k`,
  `kimi-for-coding`, and `kimi-for-coding-highspeed`, but the configured
  provider alias can vary by account. Inspect `kimi provider list` after login
  rather than assuming an alias is available everywhere. For K3 review
  invocations, set `KIMI_MODEL_THINKING_EFFORT=high` unless a task calls for
  another supported level.
- Non-interactive `kimi -p` runs under Kimi's `auto` permission policy. This
  is the intended CLI behavior: `-p` cannot be combined with a stricter
  read-only permission mode. A prompt that prohibits mutations is therefore a
  behavioral guard, not a filesystem permission boundary. For an untrusted
  review target, run Kimi in external containment such as a dedicated user or
  a container with only the intended repository mounted.
- Kimi reviewers may use web fetch, GitHub through the authenticated `gh` CLI,
  and installed skills for verification, subject to the shared review policy
  above. Do not disable Kimi's normal skill discovery when those skills are
  needed for a review.

### Setup-only authenticated smoke test

This is a launcher check, not a reviewer action. Run it before a review from
outside the worktree. It uses the already authenticated normal Kimi home; a
fresh `KIMI_CODE_HOME` has no provider configuration or OAuth credentials and
cannot successfully run a prompt.

```shell
(
  cd /tmp
  KIMI_MODEL_THINKING_EFFORT=high kimi -p "Say OK"
)
```

Do not publish raw provider JSONL, OAuth tokens, session diagnostics, metadata,
or tool output as a review result.
