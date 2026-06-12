"""
Output Formatter
================
Converts ExtractionResult into human-readable Markdown reports
and structured JSON exports.
"""

import time


class OutputFormatter:

    @staticmethod
    def to_markdown(result) -> str:
        """
        Render a complete, well-structured Markdown document from an ExtractionResult.
        """
        lines = []
        dur = _fmt_time(result.duration_seconds)

        # ── Header ──────────────────────────────
        lines += [
            f"# 📹 {result.video_title}",
            "",
            f"> **Duration:** {dur}  |  **Processed:** {result.processed_at}  "
            f"|  **Model:** {result.metadata.get('llm_model', 'N/A')}",
            "",
        ]

        # ── Summary ─────────────────────────────
        lines += [
            "## 🗒 Executive Summary",
            "",
            result.summary,
            "",
        ]

        # ── Key Topics ───────────────────────────
        if result.key_topics:
            topics_str = " · ".join(f"`{t}`" for t in result.key_topics)
            lines += [
                "## 🏷 Key Topics",
                "",
                topics_str,
                "",
            ]

        # ── Timestamps ───────────────────────────
        if result.timestamps:
            lines += ["## ⏱ Important Timestamps", ""]
            imp_order = {"high": 0, "medium": 1, "low": 2}
            sorted_ts = sorted(result.timestamps,
                               key=lambda t: t.get("time_seconds", 0))
            for ts in sorted_ts:
                imp = ts.get("importance", "medium")
                badge = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(imp, "⚪")
                label = ts.get("time_label", "??:??")
                title = ts.get("title", "")
                desc  = ts.get("description", "")
                lines.append(f"- **[{label}]** {badge} **{title}** — {desc}")
            lines.append("")

        # ── Action Items ─────────────────────────
        if result.action_items:
            lines += ["## ✅ Action Items", ""]
            priority_order = {"urgent": 0, "normal": 1, "low": 2}
            sorted_ai = sorted(result.action_items,
                               key=lambda a: priority_order.get(a.get("priority", "normal"), 1))
            for ai in sorted_ai:
                pri   = ai.get("priority", "normal")
                badge = {"urgent": "🔴", "normal": "🟡", "low": "🟢"}.get(pri, "⚪")
                task  = ai.get("task", "")
                owner = ai.get("owner")
                due   = ai.get("due_date")
                ctx   = ai.get("context", "")

                meta_parts = []
                if owner: meta_parts.append(f"Owner: **{owner}**")
                if due:   meta_parts.append(f"Due: **{due}**")
                meta_str = "  |  ".join(meta_parts)

                lines.append(f"### {badge} {task}")
                if meta_str:
                    lines.append(f"_{meta_str}_")
                if ctx:
                    lines.append(f"> _{ctx}_")
                lines.append("")

        # ── Organised Notes ───────────────────────
        if result.notes:
            lines += ["## 📚 Organised Notes", ""]
            for section in result.notes:
                lines += _render_section(section, level=3)

        # ── Transcript Preview ───────────────────
        if result.transcript_excerpt:
            lines += [
                "---",
                "## 📄 Transcript Preview",
                "",
                f"> {result.transcript_excerpt[:300]}…",
                "",
            ]

        return "\n".join(lines)


# ── Helpers ──────────────────────────────────────

def _render_section(section: dict, level: int = 3) -> list:
    lines = []
    heading = section.get("heading", "Section")
    content = section.get("content", "")
    ts_ref  = section.get("timestamp_ref")
    subs    = section.get("subsections", [])

    prefix = "#" * min(level, 6)
    ts_tag = f" _(⏱ {_fmt_time(ts_ref)})_" if ts_ref is not None else ""
    lines.append(f"{prefix} {heading}{ts_tag}")
    lines.append("")

    if content:
        lines.append(content)
        lines.append("")

    for sub in subs:
        lines += _render_section(sub, level=level + 1)

    return lines


def _fmt_time(seconds) -> str:
    if seconds is None:
        return "??"
    try:
        seconds = int(float(seconds))
    except (ValueError, TypeError):
        return "??"
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"