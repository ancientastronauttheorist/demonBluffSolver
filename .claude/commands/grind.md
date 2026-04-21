---
description: Start the autonomous ascension grinder in the background and stream per-village summaries to this chat. Ascension-complete and halts surface here (and as PushNotifications from the session).
---

Start the grinder and watch its output stream. One session plays the whole current ascension — loop semantics live inside `/play-loop`, not in the wrapper.

## What to do

1. Start the wrapper under `Monitor` with `persistent: true`. Filter to only the lines worth interrupting for:

   ```bash
   bash scripts/play_loop.sh 2>&1 | grep -E --line-buffered "VILLAGE_END|ASCENSION_COMPLETE|HALT|WRAPPER_EXIT"
   ```

   Description for the monitor: `"grind: per-village + ascension-complete + halt + wrapper exit"`.

2. Tell the user: "Grinder started. I'll post a summary after each village ends and when the ascension completes. Halts surface here."

## Handling events as they arrive

**`=== VILLAGE_END village=<N> status=<win|loss> hp=<HP> ascension=<asc_tag> ===`** — one village committed:
- Pull the tip commit title: `git log -1 --pretty=format:'%s'` (one-shot Bash). That's the village's headline.
- Post a one-liner in this chat: `village <N> · <status> · <hp>HP · <commit title>`.
- Do NOT send a `PushNotification` — the user is watching this chat.

**`=== ASCENSION_COMPLETE asc=<N> final_hp=<HP> wins=<N> losses=<count> ===`** — ascension done:
- Pull the tip commit title.
- Post: `🏆 Ascension <N> complete · <hp>HP · <wins> wins / <losses> losses · <commit title>`.
- The session already sent the `PushNotification`. Don't duplicate.

**`=== HALT reason="..." branch=<halt/...> ===`** — session halted on a bug:
- Post: `⛔ HALT · <reason> · branch <halt/...>`.
- The session already committed the fix branch and sent the `PushNotification`. Don't duplicate.
- Surface `git log -1 <branch> --pretty=format:'%s'` for context.

**`=== WRAPPER_EXIT reason=<ascension-complete|halt|crash> ... ===`** — the grind stopped:
- Tell the user why.
- If `ascension-complete`: celebrate briefly, no action needed.
- If `halt`: the halt marker above already surfaced detail.
- If `crash`: the session died without emitting a terminal marker. Propose next steps: inspect the log tail (`tail -50 logs/grind_<date>.log`), look at `git status`, decide whether to resume.
- Stop monitoring. No auto-restart — the user decides whether to begin the next ascension.

## Stopping the grind manually

If the user asks to stop: call `TaskStop` on the monitor. The underlying `claude -p` village session keeps running to whatever natural stopping point it reaches (committing the current village if possible); only the *monitor* detaches. Confirm this to the user.

## What this command does NOT do

- Play villages itself. The wrapper + headless `claude -p` session do that.
- Commit anything from this session. Every commit comes from the per-village session.
- Retry on halt or loop across ascensions. Halts and completions are always human-reviewed before starting a new ascension.
