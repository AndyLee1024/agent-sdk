"""单元测试: TotalUsageSnapshot"""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from comate_agent_sdk.llm.views import ChatInvokeUsage
from comate_agent_sdk.tokens.total_usage import LevelUsage, TotalUsageSnapshot
from comate_agent_sdk.tokens.views import TokenUsageEntry


def _make_entry(
    *,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    total_tokens: int = 150,
    cached_tokens: int = 0,
    level: str | None = "MID",
    source: str | None = "agent",
) -> TokenUsageEntry:
    return TokenUsageEntry(
        model="test-model",
        timestamp=datetime.now(),
        usage=ChatInvokeUsage(
            prompt_tokens=prompt_tokens,
            prompt_cached_tokens=cached_tokens if cached_tokens > 0 else None,
            prompt_cache_creation_tokens=None,
            prompt_image_tokens=None,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
        level=level,
        source=source,
    )


class TestLevelUsage(unittest.TestCase):
    def test_initial_state(self) -> None:
        lv = LevelUsage()
        self.assertEqual(lv.prompt_tokens, 0)
        self.assertEqual(lv.completion_tokens, 0)
        self.assertEqual(lv.total_tokens, 0)
        self.assertEqual(lv.invocations, 0)

    def test_add_entry(self) -> None:
        lv = LevelUsage()
        entry = _make_entry(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        lv._add(entry)
        self.assertEqual(lv.prompt_tokens, 100)
        self.assertEqual(lv.completion_tokens, 50)
        self.assertEqual(lv.total_tokens, 150)
        self.assertEqual(lv.invocations, 1)

    def test_serialization_roundtrip(self) -> None:
        lv = LevelUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150, cached_tokens=10, invocations=2)
        restored = LevelUsage.from_dict(lv.to_dict())
        self.assertEqual(restored.prompt_tokens, 100)
        self.assertEqual(restored.completion_tokens, 50)
        self.assertEqual(restored.total_tokens, 150)
        self.assertEqual(restored.cached_tokens, 10)
        self.assertEqual(restored.invocations, 2)


class TestTotalUsageSnapshotAddTurnEntries(unittest.TestCase):
    def test_regular_entries_grouped_by_level(self) -> None:
        snapshot = TotalUsageSnapshot.new(session_id="sess-001")
        entries = [
            _make_entry(level="MID", source="agent", total_tokens=100),
            _make_entry(level="HIGH", source="agent", total_tokens=200),
            _make_entry(level="MID", source="agent", total_tokens=150),
        ]
        snapshot.add_turn_entries(entries)

        self.assertIn("MID", snapshot.by_level)
        self.assertIn("HIGH", snapshot.by_level)
        self.assertEqual(snapshot.by_level["MID"].total_tokens, 250)
        self.assertEqual(snapshot.by_level["MID"].invocations, 2)
        self.assertEqual(snapshot.by_level["HIGH"].total_tokens, 200)
        self.assertEqual(snapshot.by_level["HIGH"].invocations, 1)

    def test_grand_total_includes_all(self) -> None:
        snapshot = TotalUsageSnapshot.new(session_id="sess-001")
        entries = [
            _make_entry(level="MID", source="agent", total_tokens=100),
            _make_entry(level="HIGH", source="agent", total_tokens=200),
        ]
        snapshot.add_turn_entries(entries)

        self.assertEqual(snapshot.grand_total.total_tokens, 300)
        self.assertEqual(snapshot.grand_total.invocations, 2)

    def test_compaction_entries_go_to_compacted_usage(self) -> None:
        snapshot = TotalUsageSnapshot.new(session_id="sess-001")
        entries = [
            _make_entry(level="MID", source="agent", total_tokens=100),
            _make_entry(level="MID", source="compaction", total_tokens=300),
        ]
        snapshot.add_turn_entries(entries)

        # compaction entry 不应在 by_level 中（除 compaction 非独立 level）
        self.assertEqual(snapshot.compacted_usage.total_tokens, 300)
        self.assertEqual(snapshot.compacted_usage.invocations, 1)
        # grand_total = agent(100) + compaction(300)
        self.assertEqual(snapshot.grand_total.total_tokens, 400)
        # by_level 只有 agent 那条
        self.assertEqual(snapshot.by_level.get("MID", LevelUsage()).total_tokens, 100)

    def test_subagent_compaction_source_detected(self) -> None:
        snapshot = TotalUsageSnapshot.new(session_id="sess-001")
        entry = _make_entry(source="subagent:researcher:compaction", total_tokens=200)
        snapshot.add_turn_entries([entry])

        self.assertEqual(snapshot.compacted_usage.total_tokens, 200)

    def test_unknown_level_entries(self) -> None:
        snapshot = TotalUsageSnapshot.new(session_id="sess-001")
        entry = _make_entry(level=None, source="agent", total_tokens=100)
        snapshot.add_turn_entries([entry])

        self.assertIn("UNKNOWN", snapshot.by_level)
        self.assertEqual(snapshot.by_level["UNKNOWN"].total_tokens, 100)

    def test_accumulates_across_multiple_calls(self) -> None:
        snapshot = TotalUsageSnapshot.new(session_id="sess-001")
        snapshot.add_turn_entries([_make_entry(level="MID", total_tokens=100)])
        snapshot.add_turn_entries([_make_entry(level="MID", total_tokens=200)])

        self.assertEqual(snapshot.by_level["MID"].total_tokens, 300)
        self.assertEqual(snapshot.grand_total.total_tokens, 300)


class TestTotalUsageSnapshotPersistence(unittest.TestCase):
    def test_save_and_load_roundtrip(self) -> None:
        snapshot = TotalUsageSnapshot.new(session_id="sess-abc")
        entries = [
            _make_entry(level="MID", source="agent", total_tokens=100),
            _make_entry(level="HIGH", source="agent", total_tokens=200),
            _make_entry(level="MID", source="compaction", total_tokens=50),
        ]
        snapshot.add_turn_entries(entries)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "total_usage.json"
            snapshot.save(path)

            restored = TotalUsageSnapshot.load(path)

        self.assertEqual(restored.session_id, "sess-abc")
        self.assertEqual(restored.by_level["MID"].total_tokens, 100)
        self.assertEqual(restored.by_level["HIGH"].total_tokens, 200)
        self.assertEqual(restored.compacted_usage.total_tokens, 50)
        self.assertEqual(restored.grand_total.total_tokens, 350)

    def test_save_creates_parent_directory(self) -> None:
        snapshot = TotalUsageSnapshot.new(session_id="sess-xyz")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "subdir" / "total_usage.json"
            snapshot.save(path)
            self.assertTrue(path.exists())

    def test_saved_json_is_human_readable(self) -> None:
        snapshot = TotalUsageSnapshot.new(session_id="sess-001")
        snapshot.add_turn_entries([_make_entry(level="MID", total_tokens=100)])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "total_usage.json"
            snapshot.save(path)
            data = json.loads(path.read_text())

        self.assertEqual(data["session_id"], "sess-001")
        self.assertIn("by_level", data)
        self.assertIn("grand_total", data)


if __name__ == "__main__":
    unittest.main()
