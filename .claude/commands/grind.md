---
description: Start the autonomous ascension grinder in the background and stream per-village summaries to this chat. Halts surface here and in a PushNotification.
---

Start the grinder and watch its output stream.

## What to do

1. Make sure we're not already grinding. Check: is `.play_loop_status` present? If yes and it's not `ok`, stop and tell the user — a previous halt needs review before a new grind.
2. Start the wrapper under `Monitor` with `persistent: true`. Filter to only the lines worth interrupting for:

   ```bash
   bash scripts/play_loop.sh 2>&1 | grep -E --line-buffered "VILLAGE_END|WRAPPER_EXIT"
   ```

   Description for the monitor: `"grind: per-village end + wrapper exit"`.

3. Tell the user: "Grinder started. I'll post a summary after each village ends. Halts surface here."

## Handling events as they arrive

Each `VILLAGE_END` line is one village completing. When you see one:

- Parse `village=<N>` and `status=<ok|halt|crash>` and `elapsed=<Xs>`.
- Pull the tip commit title: `git log -1 --pretty=format:'%s'` (one-shot Bash). That's the village's headline.
- Post a one-liner in this chat: `village <N> · <status> · <elapsed> · <commit title>`.
- Do NOT send a PushNotification for `status=ok` — the user is watching this chat.

When you see a `WRAPPER_EXIT` line:

- The grind stopped. Tell the user why (`reason=halt | crash | preflight-stale-status`).
- If `halt`: the halting village session already committed a fix branch and sent a PushNotification per the protocol. Surface the branch name (`git branch --show-current` in the last village's worktree if applicable, or `git log -1 --all --source` to find the `halt/...` branch).
- If `crash`: the wrapper is reporting the village session died without writing status. Propose next steps: inspect the log tail (`tail -50 logs/grind_<date>.log`), look at `git status`, decide whether to resume.
- Stop monitoring. No auto-restart — the user decides.

## Stopping the grind manually

If the user asks to stop: call `TaskStop` on the monitor. The already-in-flight village session keeps running to completion (it's a separate process); only the *loop* stops. Confirm this to the user.

## What this command does NOT do

- Play villages itself. The wrapper + headless `claude -p` sessions do that.
- Commit anything from this session. Every commit comes from the per-village session.
- Retry on halt. Halts are always human-reviewed before resume.
