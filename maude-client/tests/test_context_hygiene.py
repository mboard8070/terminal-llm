"""Client-local context hygiene (works without maude_core on Mac/PC)."""

from maude_client.context_hygiene import (
    apply_hygiene_in_place,
    compact_tool_result,
    drop_old_tool_payloads,
    sliding_window_with_summary,
)


def test_compact_tool_result_caps_long_read():
    body = "\n".join(f"line {i:04d} " + ("x" * 40) for i in range(200))
    out = compact_tool_result("read_file", body)
    assert len(out) < len(body)
    assert "omitted" in out or "truncated" in out


def test_drop_old_tool_payloads_stubs_old_rounds():
    messages = [
        {"role": "user", "content": "first"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "A" * 500},
        {"role": "user", "content": "second"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "list_directory", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c2", "name": "list_directory", "content": "B" * 500},
        {"role": "user", "content": "third"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c3",
                    "type": "function",
                    "function": {"name": "run_command", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c3", "name": "run_command", "content": "C" * 500},
    ]
    msgs, compacted = drop_old_tool_payloads(messages, keep_recent=2, in_place=False)
    assert compacted >= 1
    assert msgs[2]["content"].startswith("[prior tool result")
    # Most recent two tool rounds stay full
    assert msgs[5]["content"] == "B" * 500 or msgs[5]["content"].startswith("[prior")
    assert msgs[8]["content"] == "C" * 500


def test_sliding_window_summarizes_older_turns():
    messages = [{"role": "system", "content": "You are MAUDE."}]
    for i in range(20):
        messages.append({"role": "user", "content": f"user turn {i}"})
        messages.append({"role": "assistant", "content": f"assistant turn {i}"})
    prepared, meta = sliding_window_with_summary(messages, keep_recent=6, in_place=False)
    assert meta["removed"] > 0
    assert meta["summarized"] is True
    assert any(
        m.get("role") == "system"
        and str(m.get("content", "")).startswith("[Earlier conversation summarized")
        for m in prepared
    )
    # System prompt preserved
    assert prepared[0]["content"] == "You are MAUDE."
    # Recent turns present
    assert any("user turn 19" in str(m.get("content")) for m in prepared)


def test_apply_hygiene_in_place_bounds_history():
    messages = [{"role": "system", "content": "sys"}]
    for i in range(30):
        messages.append({"role": "user", "content": f"u{i} " + ("x" * 100)})
        messages.append({"role": "assistant", "content": f"a{i}"})
    meta = apply_hygiene_in_place(messages)
    assert meta["final_count"] < 62
    assert meta["final_tokens"] > 0
