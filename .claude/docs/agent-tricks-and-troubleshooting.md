# Agent tricks and troubleshooting

Read this document before invoking Claude CLI or Codex CLI from another agent.
It contains local invocation details that are easy to get wrong, especially for
non-interactive Claude review runs.

This note covers practical ways to use Codex CLI and Claude CLI as local engineering agents, especially when one agent is used to review or debug the other agent's work.

## Codex CLI

Codex is useful for local repository work where the agent should inspect files, edit code, run tests, and keep working until a task is complete.

Common commands:

```shell
codex
codex "Explain this codebase"
codex exec "Review the current diff for correctness bugs"
codex exec --json "Summarise the failing tests"
codex review
codex doctor
codex resume --last
codex mcp list
codex plugin list
```

Useful capabilities:

- Interactive terminal UI for iterative coding and review.
- Non-interactive automation with `codex exec`.
- Local code review with `codex review` or `/review` inside the interactive UI.
- Git-aware operation over the current worktree.
- Sandbox and approval controls with `--sandbox` and `--ask-for-approval`.
- Machine-readable automation output with `codex exec --json`.
- MCP server management with `codex mcp`.
- Plugin management with `codex plugin`.
- Diagnostic checks with `codex doctor`.
- Session continuation with `codex resume`.

Recommended local patterns:

```shell
# Ask for a bounded review of local changes.
codex exec "Review uncommitted changes for correctness bugs only"

# Pass logs as context while keeping the prompt explicit.
poetry run pytest tests/foo.py -q 2>&1 \
  | codex exec "Explain the failure and suggest the smallest fix"

# Run with explicit permissions in automation. `codex exec` selects the sandbox
# directly and does not take `--ask-for-approval` (that is an interactive flag).
codex exec --sandbox workspace-write "Fix the failing focused test"

# Debug local setup.
codex doctor
```

Use `codex exec` for CI-like or scripted work. It streams progress to stderr and final output to stdout, which makes it easier to pipe the result into files or other commands.

Use interactive `codex` when the task needs back-and-forth decisions, screenshots, manual inspection, or careful approval of edits.

### Always run Codex reviews in streaming mode

Run every non-interactive Codex review (plan review, code review, sanity check)
in streaming mode with `--json`. Plain text mode (`codex exec "..."`) only emits
the final answer once the model has finished the entire review, and any
`| tail`, `| head`, or capture-to-file buffers that final block until the pipe
closes. When the run is backgrounded or captured, the output file then stays
**0 bytes** until completion — indistinguishable from a hang, and you cannot see
progress or interim tool calls.

`--json` instead emits a JSONL event stream (reasoning, tool calls, and the final
message) line-by-line as they happen, so a backgrounded run's file grows live and
can be tailed for progress.

```shell
# Streaming read-only review. Note: DO NOT pipe through `tail`/`head` — that
# reintroduces buffering. Write the raw JSONL stream to a file instead.
codex exec --json --sandbox read-only \
  "Review the uncommitted diff for correctness bugs only. Findings first with file:line." \
  > /tmp/codex-review.jsonl

# Follow progress live from another step (or when backgrounded):
tail -f /tmp/codex-review.jsonl        # interactive shells only

# Extract just the final assistant message from the JSONL when done:
#   each line is a JSON event; the final answer is the last agent/message event.
```

When backgrounding a Codex review, always use `--json` and read the raw output
file for interim events. If you instead run text mode in the background, the file
will look empty (0 bytes) the whole time and you will not be able to tell a slow
review from a stuck one.

Redirect stdin from `/dev/null` for background/non-interactive runs. With an open
stdin pipe, `codex exec` prints `Reading additional input from stdin...` and waits
for EOF (it appends piped stdin as a `<stdin>` block), so the run stalls forever
even though the prompt was passed as an argument. Always append `< /dev/null`:

```shell
codex exec --json --sandbox read-only "…prompt…" < /dev/null > /tmp/codex-review.jsonl
```

Approval flags: `codex exec` does **not** accept `--ask-for-approval` (that flag
belongs to interactive `codex`). For non-interactive review runs pick the sandbox
directly — `--sandbox read-only` needs no approval and is the correct choice for
reviews. Use `--sandbox workspace-write` only when the run must edit files.

```shell
# Correct non-interactive review invocation (streaming, read-only, no approval flag).
codex exec --json --sandbox read-only "Review uncommitted changes for correctness bugs only" \
  > /tmp/codex-review.jsonl
```

## Claude CLI

Claude CLI is useful for independent second opinions, code reviews, background agents, and checking whether another agent's change makes sense.

Common commands:

```shell
claude
claude -p "Review the current worktree diff"
claude -p "Review the current worktree diff" --output-format stream-json --verbose
claude auth status
claude ultrareview master --timeout 15
claude doctor
claude agents
claude mcp
claude plugin list
claude --help
```

Useful capabilities:

- Interactive Claude Code session by default.
- Non-interactive print mode with `-p` / `--print`.
- Streaming automation output with `--output-format stream-json`.
- Tool restrictions with `--allowedTools` and `--disallowedTools`.
- Permission mode control with `--permission-mode`.
- Custom or selected agents with `--agent` and `--agents`.
- Cloud-hosted multi-agent review with `claude ultrareview`.
- Safe mode for debugging broken customisations with `--safe-mode`.
- Bare mode for minimal startup with `--bare`.
- Debug logging with `--debug` or `--debug-file`.
- MCP and plugin management.

Important authentication note:

- Do not use `--bare` to check whether Claude is signed in. Bare mode skips
  keychain/OAuth credentials by design and only uses `ANTHROPIC_API_KEY` or an
  `apiKeyHelper` from `--settings`. A signed-in Unix user can therefore see
  `Not logged in` in `--bare` mode even though normal `claude -p` works.
- Use normal mode for local signed-in accounts:

```shell
claude auth status
claude -p "Say OK"
```

- Use `--safe-mode` instead of `--bare` when debugging broken customisations
  but you still want normal auth, model selection and built-in permissions to
  work.

Recommended local patterns:

```shell
# Smoke-test non-interactive auth and startup.
claude -p "Say OK"

# Plain one-shot review. Good when you can wait for final buffered output.
claude -p "Review the current git diff for correctness bugs"

# Better for long reviews: stream progress and tool calls.
claude -p "Review the current git diff for correctness bugs" \
  --output-format stream-json \
  --verbose

# Restrict tools for a read-only review.
claude -p "Review the current worktree diff. Do not edit files." \
  --permission-mode bypassPermissions \
  --dangerously-skip-permissions \
  --allowedTools "Bash,Read,Grep,Glob"

# Safer read-only review without broad bypass mode.
claude -p "Review the current worktree diff. Do not edit files. Findings first." \
  --permission-mode dontAsk \
  --allowedTools "Bash(git status:*),Bash(git diff:*),Bash(sed:*),Bash(rg:*)"

# Avoid pasting huge diffs into the prompt. Make Claude inspect files itself.
claude -p "Review uncommitted changes. First run git diff --name-only, then inspect targeted diffs."

# Run cloud review when account credits and PR/base context are available.
claude ultrareview master --timeout 15
```

For long-running `claude -p` jobs, prefer `--output-format stream-json --verbose`. Text mode can look idle because useful output may be buffered until the final answer.

Use a 15-minute wall-clock deadline for an external-agent process, including
Claude CLI and Codex CLI (for example, `timeout 900 claude -p ...`). This gives
a legitimate review enough time to inspect the worktree; it does not override
the separate one-minute no-output rule for a grounded Claude review.

### Do not mistake an incomplete stream poll for an Opus 5 failure

For `claude -p --output-format stream-json`, an assistant/tool event is not a
review result. Opus 5 can spend several minutes gathering repository context
and emit many tool events before it writes the final `result` event. In some
agent-runner shells, the foreground command wrapper can also return before its
redirected child has finished writing the JSONL file. Therefore, do **not**
conclude that Claude "stopped mid-inspection" merely because a short poll has
no final text, a shell cell reports completion, or the last JSONL line is a
`thinking`/`tool_use` event.

A Claude review is complete only when its JSONL has a final event with all of:

- `type == "result"`
- `is_error == false`
- `stop_reason == "end_turn"`
- `terminal_reason == "completed"`
- a non-empty `result`

`rate_limit_event` is telemetry, not necessarily a failure. In particular,
`status == "allowed"` can appear before a successful final result. Check the
final `result` event before reporting a rate-limit failure.

Start a long Opus review detached, retain its PID and raw files, then poll the
process and JSONL rather than repeatedly launching new reviews. `--model opus`
selects the current Opus alias (currently reported as `claude-opus-5` in the
result metadata).

```shell
review_file="/tmp/claude-review-$RANDOM.jsonl"
review_err="/tmp/claude-review-$RANDOM.err"

nohup timeout 900 claude -p "Review the current uncommitted diff read-only. Do not edit files or run tests. Return findings with file:line references." \
  --model opus \
  --output-format stream-json --verbose \
  --permission-mode dontAsk \
  --allowedTools "Bash(git status:*),Bash(git diff:*),Bash(sed:*),Bash(rg:*),Bash(grep:*),Bash(ls:*),Read,Grep,Glob" \
  --no-session-persistence \
  < /dev/null > "$review_file" 2> "$review_err" &
review_pid=$!
echo "Claude review PID: $review_pid"
```

Poll it from a later command. Do not start another review while the process is
alive or the JSONL has no terminal `result` event:

```shell
ps -p "$review_pid" -o pid=,stat=,etime=,cmd=
wc -c "$review_file" "$review_err"
tail -n 20 "$review_file"

python3 -c '
import json
from pathlib import Path

events = [json.loads(line) for line in Path("'$review_file'").read_text().splitlines()]
result = next((event for event in reversed(events) if event.get("type") == "result"), None)
assert result, "Review is still incomplete: no final result event"
assert not result["is_error"]
assert result["stop_reason"] == "end_turn"
assert result["terminal_reason"] == "completed"
print(result["result"])
'
```

If the process has exited and there is no valid final `result`, inspect the raw
stderr and any `permission_denials` included in a partial result before retrying.
Only then use a narrower prompt or an inline no-tools review. Re-running an
unfinished-looking Opus review without this check wastes quota and can create a
real rate-limit failure.

If a broad review stalls, first verify that basic non-interactive mode and
read-only Bash tools work before assuming auth is broken:

```shell
claude -p "Say OK"
claude -p "Run git status --short and summarise it in one sentence." \
  --allowedTools "Bash(git status:*)"
```

If these work but the broad review times out, shrink the request: ask Claude to
inspect `git diff --name-only` first, review one file group at a time, or provide
a concise summary of the proposed fix instead of embedding a large diff.

### Reviewing a plan or document with Claude CLI

For Markdown plan reviews, default to a bounded no-tools Fable review after
the relevant code has already been inspected by the primary agent. Do not
start with a grounded tool-using Claude review for a simple plan re-review: it
can spend time on repository inspection that does not improve the review.

In an agent runner, **do not use a foreground text-mode `claude -p` command**
for this. The runner can detach or buffer its child process, leaving no
reliable final answer. Always write a streaming JSONL result to a known file,
retain the PID, and validate the terminal event. The following is the standard
Fable plan-review recipe:

```shell
review_file="/tmp/claude-fable-plan-$RANDOM.jsonl"
review_err="/tmp/claude-fable-plan-$RANDOM.err"
plan_file=".claude/plans/my-plan.md"

nohup timeout 900 claude -p "$(sed '1iDo not use tools. Review only the plan text below. Do not overengineer. Return concise actionable findings, or say no blocking findings.\n' "$plan_file")" \
  --model fable \
  --tools "" \
  --output-format stream-json --verbose \
  --permission-mode dontAsk \
  --no-session-persistence \
  --max-budget-usd 1 \
  < /dev/null > "$review_file" 2> "$review_err" &
review_pid=$!
printf 'Claude Fable review PID: %s\\n' "$review_pid"
```

Poll the same run; do not launch another review while it is still alive:

```shell
ps -p "$review_pid" -o pid=,stat=,etime=,cmd=
wc -c "$review_file" "$review_err"
tail -n 20 "$review_file"
```

When it exits, validate the result before reading or reporting it:

```shell
python3 -c '
import json
import sys
from pathlib import Path

events = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
result = next((event for event in reversed(events) if event.get("type") == "result"), None)
assert result, "Incomplete review: no final result event"
assert result.get("is_error") is False, result
assert result.get("stop_reason") == "end_turn", result
assert result.get("terminal_reason") == "completed", result
assert result.get("result"), "Incomplete review: empty result"
print(result["result"])
' "$review_file"
```

Use the same recipe for a final blocking-only pass, changing only the first
prompt sentence to `Return only blocking findings, or say no blocking
findings.` Do not invent a new invocation pattern for the second pass.

Use a 15-minute deadline for a Fable review. A review that is continuously
writing new JSONL events is making progress and must be allowed to finish;
do not confuse its elapsed time with a hang. Only stop it early when the JSONL
file and stderr have shown no growth for roughly one minute, or when it is
waiting for an unavailable permission prompt.

If the run reaches its 15-minute deadline, or exits without a terminal result,
inspect `"$review_err"` and report that the Claude review was incomplete.
Do not call that outcome “no findings”, do not infer a review from partial
tool/thinking events, and do not automatically retry it. A human or primary
agent can then choose either a smaller excerpt or a focused grounded review.

Only use a grounded repository review when Claude specifically needs fresh code
inspection, for example when the primary agent has not checked the relevant
files or when the plan makes claims that need independent verification against
the worktree. In that case, allow only read-only tools and make the scope
explicit:

```shell
claude -p "Review .claude/plans/my-plan.md for correctness and completeness. Focus on implementation risks, missing code paths, and test gaps. Keep the review concise and actionable." \
  --allowedTools Read,Grep,Glob,Bash \
  --permission-mode dontAsk \
  --output-format stream-json \
  --verbose
```

If a grounded review produces no output after roughly a minute, stop it and
switch to the no-tools inline review unless fresh repository inspection is
strictly required.

Notes:

- Use `--tools ""` only when the prompt embeds all necessary context.
- If you allow tools, use comma-separated tool names for `--allowedTools`.
- `--permission-mode dontAsk` avoids interactive permission prompts in
  non-interactive review runs.
- `--no-session-persistence` keeps one-off reviews from polluting later
  `claude --continue` sessions.
- `--max-budget-usd` is optional but useful for bounded document reviews.

## Cross-agent review patterns

Use the other agent as a reviewer when:

1. The change touches state accounting, execution, security, or money movement.
2. The first agent wrote a large test or complex fixture.
3. You want a second model to challenge assumptions before opening a pull request.
4. You suspect the first agent is stuck in a local optimum.

Good review prompt:

```text
Review the current uncommitted worktree diff for correctness bugs only.
Do not run the full test suite.
Do not paste the full git diff into context.
First inspect git status --short, git diff --name-only, and targeted diffs.
Focus on behavioural regressions and test fragility.
Return findings first with file:line references.
If there are no high-confidence bugs, say so clearly and list residual risks.
```

Avoid asking for broad "thoughts" on a large diff. Ask for a scoped review:

- correctness bugs
- behavioural regressions
- missing tests
- test fragility
- security or money-movement risks

## Grok CLI

Grok is an independent second opinion for plans, diffs and pull requests. Its
headless mode is `-p` / `--single`. Verify the installed flags before changing
these commands: releases have changed their review and permission behaviour.

```shell
grok --help
grok --version
```

### How to run a Grok review

Use `--output-format streaming-json` for every non-trivial review. It writes
JSONL events while Grok reasons, calls tools and produces its answer; plain JSON
only appears when the run ends and can look indistinguishable from a hang. Write
the raw stream to a file. Do not pipe the review through `head` or `tail`.

For a repository-grounded review, start a detached process with a 15-minute
deadline. Grok's current headless CLI can cancel after its first tool request
under `dontAsk`; a grounded review therefore requires
`bypassPermissions --always-approve`. This grants the process broad authority,
so use it only in a trusted repository, disable web search, forbid edits in the
prompt, and never combine it with an untrusted prompt or repository.

```shell
# Write the prompt with a normal file-editing tool first.
nohup timeout 900 grok --prompt-file /tmp/grok-pr-review.md \
  --model grok-4.5 \
  --reasoning-effort high \
  --permission-mode bypassPermissions --always-approve \
  --disable-web-search --no-plan --no-subagents --max-turns 48 \
  --output-format streaming-json \
  --cwd "$(pwd)" \
  < /dev/null > /tmp/grok-review.jsonl 2>/tmp/grok-review.err &
review_pid=$!
```

Use a focused prompt such as:

```text
Review the current pull request read-only. Do not edit files, change GitHub
state, or run tests. First inspect the PR metadata and changed-file list, then
inspect only the relevant source and test hunks. Focus on correctness bugs,
behavioural regressions, missing tests, security or money-movement risks, and
repository-instruction compliance. Return findings first with file:line
references, then residual risks.
```

For GitHub PRs, tell Grok to use `gh pr view` and `gh pr diff` rather than a
stale local branch. Do not ask it to fetch an unfiltered full diff. Exclude
generated artefacts, lock files and large narrative plans from the initial pass;
review security-relevant generated output separately. If it reaches its turn
limit, split the changed files into review groups instead of raising the limit.

### Validate the result

Poll the detached process and raw files from a later command. A review is valid
only when the final JSONL event is `type == "end"` and
`stopReason == "EndTurn"`:

```shell
ps -p "$review_pid" -o pid=,stat=,etime=,cmd=
wc -c /tmp/grok-review.jsonl /tmp/grok-review.err
tail -n 30 /tmp/grok-review.jsonl
python3 -c "import json; events=[json.loads(line) for line in open('/tmp/grok-review.jsonl')]; end=events[-1]; assert end['type'] == 'end' and end['stopReason'] == 'EndTurn'; print(''.join(event['data'] for event in events if event['type'] == 'text'))"
```

Treat `Cancelled`, `PermissionCancelled`, an absent `end` event, a blank raw
file, and `max turns reached` as incomplete reviews, never as no findings. A
short preamble followed by cancellation is not a review result. Inspect stderr,
the process state and a small smoke test before retrying; do not automatically
run `grok update` while diagnosing a failure.

### No-tools fallback

If grounded mode fails, put only the highest-risk, self-contained excerpts in a
prompt file and run a no-tools review. Keep the excerpts small: an inline full
diff can exhaust the context or produce findings based on missing surrounding
code.

```shell
timeout 240 grok -p "$(cat /tmp/grok-notools-review.txt)" \
  --model grok-4.5 --no-plan --no-subagents --tools '' \
  --disable-web-search --output-format streaming-json --cwd "$(pwd)" \
  < /dev/null > /tmp/grok-notools-review.jsonl 2>/tmp/grok-notools-review.err
```

Start this prompt with `NO TOOLS — review only the code below.` It must include
the changed function and enough of the caller or guard to establish control
flow. Validate `EndTurn` for this output too.

### Review rules and gotchas

- Smoke-test the CLI before a costly review. `grok models` alone is not a
  reliable authentication check; require a non-empty streaming result ending in
  `EndTurn` from `grok -p 'Reply with exactly: OK. Do not use tools.'`.
- Do not use partial tool allow-lists for grounded PR review: they can remove a
  supporting inspection tool and cause an opaque cancellation. Conversely, do
  not use broad permission flags for a no-tools review.
- Verify every finding against the actual file and cited line before changing
  code. Incomplete runs and no-tools excerpts can hallucinate paths or infer a
  defect from omitted context.
- Confirm `--cwd` identifies the intended worktree and that the reviewed diff
  is non-empty before trusting a no-findings result.
- Remove `/tmp/grok-*.jsonl`, error logs and prompt files when the review is
  complete.

## Common failure modes

### The command looks hung

Symptoms:

- `claude -p` prints nothing for a long time.
- The terminal appears idle, but the process is still alive.

Causes:

- Text output is buffered until the final answer.
- The model is doing a long review or reading a large diff.
- The prompt caused the agent to paste a huge diff into context.
- A subprocess is waiting for input or a permission decision.

Avoid it:

```shell
claude -p "Review the current diff" --output-format stream-json --verbose
```

For Codex automation, use:

```shell
codex exec --json "Review the current diff"
```

Also constrain the prompt:

```text
Do not paste the full diff into context. Use git diff --name-only first, then inspect targeted hunks.
```

### The review consumes too much context

Symptoms:

- The model reads `git diff` for a large change and slows down.
- Output includes truncated tool results.
- The final answer misses important details.

Avoid it:

- Start with `git diff --stat` and `git diff --name-only`.
- Inspect changed files with `sed`, `nl`, `rg`, or targeted `git diff -- path`.
- Ask the reviewer to avoid full diff dumps.
- Split reviews by topic or file group.

Better prompt:

```text
Review only tradeexecutor/cli/testtrade.py and tradeexecutor/strategy/pandas_trader/position_manager.py first.
Then inspect tests only if needed to validate coverage.
```

### The agent cannot use tools

Symptoms:

- Claude says it cannot inspect files.
- Codex refuses to edit or run commands.
- A non-interactive run exits after a permission problem.

Avoid it:

- For Codex, set the sandbox explicitly (`codex exec` selects the sandbox
  directly; it has no `--ask-for-approval` flag):

```shell
codex exec --sandbox workspace-write "Run the focused test and fix failures"
```

- For Claude, set explicit permission mode and allowed tools:

```shell
claude -p "Read-only review" \
  --permission-mode bypassPermissions \
  --dangerously-skip-permissions \
  --allowedTools "Bash,Read,Grep,Glob"
```

Use broad bypass modes only in trusted repositories or externally sandboxed environments.

### The cloud review does not start

Symptoms:

- `claude ultrareview` exits immediately.
- Error mentions usage credits or account limits.

Example:

```text
Ultrareview could not launch: Usage credits exhausted.
```

Avoid it:

- Fall back to local `claude -p`.
- Use streaming output for visibility.
- Narrow the review prompt to reduce cost.
- Run local focused tests yourself and include the results in the final assessment.

### The agent reviews the wrong tree

Symptoms:

- Findings refer to the parent repository instead of the worktree.
- Tests import parent source instead of worktree source.
- The branch name or status does not match expectations.

Avoid it:

```shell
pwd
git status --short --branch
git rev-parse --show-toplevel
```

For this repository's worktrees, run tests through the parent Poetry environment but force worktree imports:

```shell
source .local-test.env && PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest tests/path/to/test.py
```

### The agent misses repository instructions

Symptoms:

- Test docstrings do not follow repo rules.
- Commands omit `source .local-test.env`.
- Python style diverges from `AGENTS.md`.

Avoid it:

- Tell the reviewer to read `AGENTS.md` first.
- Cite the relevant instruction in the prompt.
- Ask specifically for "AGENTS.md compliance" as a review axis.

Good prompt:

```text
Read AGENTS.md first. Review the new pytest tests for repository instruction compliance:
docstring format, comments matching steps, type hints, and command invocation assumptions.
```

### The agent runs too much

Symptoms:

- Full test suite starts unexpectedly.
- Long-running fork tests or Docker pulls start during review.
- CI-like commands exceed local time budgets.

Avoid it:

- Say "do not run the full test suite".
- Name the exact tests that may be run.
- For review-only work, restrict tools to read-only commands.

Example:

```text
Do not run tests. Inspect the code and tell me what focused tests should be run.
```

### The agent changes files during a review

Symptoms:

- A review command edits files.
- Formatting or unrelated cleanup appears in `git diff`.

Avoid it:

- For Claude, omit edit tools from `--allowedTools`.
- For Codex, ask for review only and use read-only sandbox:

```shell
codex exec --sandbox read-only "Review uncommitted changes for correctness bugs"
```

### Output is not machine-readable

Symptoms:

- Scripts cannot reliably parse the answer.
- Progress messages are mixed with final output.

Avoid it:

- Codex: use `codex exec --json` for JSONL event streams.
- Claude: use `claude -p --output-format json` for one result or `stream-json` for live events.
- Ask for a schema when stable fields are needed.

Claude example:

```shell
claude -p "Return {\"findings\": [...], \"risk\": \"...\"}" \
  --json-schema '{"type":"object","properties":{"findings":{"type":"array"},"risk":{"type":"string"}},"required":["findings","risk"]}'
```

### Codex rejects the configured model

Symptoms:

- `codex exec` fails immediately with
  `The '<model>' model requires a newer version of Codex. Please upgrade...`
  (streamed as a `turn.failed` JSONL event).
- Overriding with `-m <other-model>` fails with
  `model is not supported when using Codex with a ChatGPT account` for every
  alternative you try.

Cause: the standalone Codex install is older than the model pinned in
`~/.codex/config.toml`, and ChatGPT accounts cannot fall back to older models.

Fix: self-update the standalone install, then retry — no config change needed:

```shell
codex update
codex exec --json --sandbox read-only "Say OK and nothing else." < /dev/null
```

Some models are gated by **account entitlement**, not by CLI version, and
updating will not unlock them. Observed 2026-07-25 with a ChatGPT account:

```text
The 'sol' model is not supported when using Codex with a ChatGPT account.
```

`codex update` (0.144.4 → 0.145.0) did not change this, and the `sol-preview`,
`gpt-5-sol` and `solaris` spellings were rejected the same way, while the
default model answered normally. Distinguish the two cases before spending time
on upgrades: a *version* problem says "requires a newer version of Codex", an
*entitlement* problem says "not supported when using Codex with a ChatGPT
account". For the latter, either use a different account/auth method or fall
back to the default model and state in the review write-up which model actually
ran.

### Authentication or MCP setup is broken

Symptoms:

- `doctor` reports missing auth.
- MCP servers show `needs-auth`.
- Tools that depend on external services are absent.
- `claude --bare -p "Say OK"` says `Not logged in`.

Avoid it:

```shell
codex doctor
codex mcp list
claude auth status
claude -p "Say OK"
claude doctor
claude mcp
```

Do not diagnose normal Claude CLI auth with `--bare`; it intentionally skips
keychain/OAuth credentials. Use `claude auth status` and a normal `claude -p`
smoke test instead.

Do not assume missing MCP tools are model limitations. Check installation, auth, workspace policy, and whether the session needs restarting after a config change.

## Practical checklist

Before launching another agent:

1. Confirm the working directory and branch.
2. Decide whether the task is interactive, non-interactive, or cloud review.
3. Restrict tools if it is a review.
4. Prefer streaming JSON for long non-interactive jobs.
5. Tell the agent not to paste huge diffs.
6. Name the exact risk areas to review.
7. Ask for file:line findings and residual risks.
8. Run focused tests yourself when the reviewer cannot.

After the agent finishes:

1. Separate high-confidence findings from speculation.
2. Verify any proposed bug against the code.
3. Apply only fixes that match the original task.
4. Re-run focused tests if code changed.
5. Record useful failure modes in this document.

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
outside the worktree. It uses the already authenticated normal Kimi home. A
fresh `KIMI_CODE_HOME` normally has no provider configuration or OAuth
credentials; configure a provider explicitly (for example with the applicable
`KIMI_MODEL_*` variables) before attempting this check from a fresh home.

```shell
(
  cd /tmp
  KIMI_MODEL_THINKING_EFFORT=high kimi -p "Say OK"
)
```

Do not publish raw provider JSONL, OAuth tokens, session diagnostics, metadata,
or tool output as a review result.
