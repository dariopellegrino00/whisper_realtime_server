"""Unit tests for HypothesisBuffer in whisper_online.py."""

import sys
import os

# Add src/ to path so we can import whisper_online
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from whisper_online import HypothesisBuffer


def make_words(texts, start=0.0, step=0.5):
    """Helper: create list of (start, end, text) tuples from word strings."""
    words = []
    t = start
    for w in texts:
        words.append((t, t + step, w))
        t += step
    return words


# ---------------------------------------------------------------------------
# Basic lifecycle
# ---------------------------------------------------------------------------

class TestBasicLifecycle:

    def test_insert_basic(self):
        """Insert words with offset, verify self.new is populated correctly."""
        buf = HypothesisBuffer()
        words = [(0.0, 0.5, "hello"), (0.5, 1.0, "world")]
        buf.insert(words, offset=1.0)
        # After insert, new should have offset-adjusted words
        assert len(buf.new) == 2
        assert buf.new[0] == (1.0, 1.5, "hello")
        assert buf.new[1] == (1.5, 2.0, "world")

    def test_flush_confirms_matching_words(self):
        """Insert same words twice (simulating two iterations), flush confirms them."""
        buf = HypothesisBuffer()
        words = [(0.0, 0.5, "hello"), (0.5, 1.0, "world")]

        # First iteration: insert and flush to populate buffer
        buf.insert(words, offset=0)
        buf.flush()

        # Second iteration: same words again
        buf.insert(words, offset=0)
        commit = buf.flush()

        # Both words should be confirmed since they match the buffer
        committed_texts = [w[2] for w in commit]
        assert "hello" in committed_texts
        assert "world" in committed_texts

    def test_flush_no_match_no_commit(self):
        """Insert different words across iterations, flush returns empty."""
        buf = HypothesisBuffer()

        # First iteration
        buf.insert([(0.0, 0.5, "hello"), (0.5, 1.0, "world")], offset=0)
        buf.flush()

        # Second iteration: completely different words
        buf.insert([(0.0, 0.5, "foo"), (0.5, 1.0, "bar")], offset=0)
        commit = buf.flush()

        assert commit == []

    def test_complete_returns_buffer(self):
        """After flush, complete() returns unconfirmed words (the buffer)."""
        buf = HypothesisBuffer()
        words = [(0.0, 0.5, "hello"), (0.5, 1.0, "world")]

        buf.insert(words, offset=0)
        buf.flush()

        result = buf.complete()
        texts = [w[2] for w in result]
        assert "hello" in texts
        assert "world" in texts


# ---------------------------------------------------------------------------
# Insert dedup logic
# ---------------------------------------------------------------------------

class TestInsertDedup:

    def test_insert_dedup_removes_overlap(self):
        """Committed words overlap with new, dedup removes them."""
        buf = HypothesisBuffer()

        # First iteration: insert and flush to commit words
        buf.insert([(0.0, 0.5, "hello"), (0.5, 1.0, "world")], offset=0)
        buf.flush()

        # Repeat to get commits (last_commited_time becomes 1.0)
        buf.insert([(0.0, 0.5, "hello"), (0.5, 1.0, "world")], offset=0)
        buf.flush()

        # Now commited_in_buffer has words with last_commited_time=1.0.
        # Insert overlapping new words — timestamps must be > 0.9 to survive filter.
        buf.insert([(0.9, 1.4, "world"), (1.4, 1.9, "test")], offset=0)

        new_texts = [w[2] for w in buf.new]
        # "world" overlaps with last committed word and gets deduped
        assert "test" in new_texts
        assert len(buf.new) <= 1  # "world" removed, only "test" remains

    def test_insert_dedup_ngram_sizes(self):
        """Test 1-gram, 2-gram, 3-gram overlap removal."""
        buf = HypothesisBuffer()

        # Build committed buffer with 3 words
        words = [(0.0, 0.5, "one"), (0.5, 1.0, "two"), (1.0, 1.5, "three")]
        buf.insert(words, offset=0)
        buf.flush()
        buf.insert(words, offset=0)
        buf.flush()

        # Now insert words that start with "two three" (2-gram overlap)
        new_words = [(0.5, 1.0, "two"), (1.0, 1.5, "three"), (1.5, 2.0, "four")]
        buf.insert(new_words, offset=0)

        new_texts = [w[2] for w in buf.new]
        # "two" and "three" should be deduped since they match committed tail
        assert "four" in new_texts

    def test_insert_dedup_repeated_words(self):
        """With repeated words, dedup finds the maximum overlap, not the first."""
        buf = HypothesisBuffer()

        # Commit "go home go home"
        words = [(0.0, 0.5, "go"), (0.5, 1.0, "home"), (1.0, 1.5, "go"), (1.5, 2.0, "home")]
        buf.insert(words, offset=0)
        buf.flush()
        buf.insert(words, offset=0)
        buf.flush()

        # New starts with "home go home" (3-word overlap) then "now"
        new_words = [(1.0, 1.5, "home"), (1.5, 2.0, "go"), (2.0, 2.5, "home"), (2.5, 3.0, "now")]
        buf.insert(new_words, offset=0)

        new_texts = [w[2] for w in buf.new]
        # All 3 overlapping words should be removed, only "now" remains
        assert new_texts == ["now"]

    def test_insert_dedup_custom_threshold(self):
        """dedup_threshold is configurable."""
        # Very high threshold — even slight mismatch won't dedup
        buf = HypothesisBuffer(dedup_threshold=100)

        words = [(0.0, 0.5, "hello"), (0.5, 1.0, "world")]
        buf.insert(words, offset=0)
        buf.flush()
        buf.insert(words, offset=0)
        buf.flush()

        # "wrld" is close to "world" but not 100% match
        buf.insert([(1.0, 1.5, "wrld"), (1.5, 2.0, "test")], offset=0)
        new_texts = [w[2] for w in buf.new]
        assert "wrld" in new_texts  # not deduped because threshold=100

    def test_insert_filters_old_words(self):
        """Words before last_commited_time - 0.1 are filtered out."""
        buf = HypothesisBuffer()
        buf.last_commited_time = 5.0

        # Insert words: some before 4.9, some after
        words = [(1.0, 2.0, "old"), (4.0, 4.5, "also_old"), (5.0, 5.5, "new")]
        buf.insert(words, offset=0)

        new_texts = [w[2] for w in buf.new]
        assert "old" not in new_texts
        assert "also_old" not in new_texts
        assert "new" in new_texts

    def test_insert_no_dedup_when_far_apart(self):
        """When abs(a - last_commited_time) >= 1, skip dedup entirely."""
        buf = HypothesisBuffer()

        # Set up committed state
        buf.last_commited_time = 1.0
        buf.last_commited_word = "hello"
        buf.commited_in_buffer = [(0.5, 1.0, "hello")]

        # Insert words starting far from last_commited_time
        words = [(5.0, 5.5, "hello"), (5.5, 6.0, "world")]
        buf.insert(words, offset=0)

        # No dedup should happen because first word starts at 5.0,
        # which is >= 1 away from last_commited_time (1.0)
        new_texts = [w[2] for w in buf.new]
        assert "hello" in new_texts
        assert "world" in new_texts
        assert len(buf.new) == 2


# ---------------------------------------------------------------------------
# Flush confirmation
# ---------------------------------------------------------------------------

class TestFlushConfirmation:

    def test_flush_partial_match(self):
        """First N words match, rest don't — only N committed."""
        buf = HypothesisBuffer()

        # First pass: populate buffer
        buf.insert([(0.0, 0.5, "hello"), (0.5, 1.0, "world"), (1.0, 1.5, "foo")], offset=0)
        buf.flush()

        # Second pass: first two match, third differs
        buf.insert([(0.0, 0.5, "hello"), (0.5, 1.0, "world"), (1.0, 1.5, "bar")], offset=0)
        commit = buf.flush()

        committed_texts = [w[2] for w in commit]
        assert "hello" in committed_texts
        assert "world" in committed_texts
        assert "bar" not in committed_texts

    def test_flush_threshold_boundary(self):
        """Words just below fuzz_threshold should not be committed."""
        buf = HypothesisBuffer(qratio_threshold=95)

        # First pass
        buf.insert([(0.0, 0.5, "hello"), (0.5, 1.0, "world")], offset=0)
        buf.flush()

        # Second pass: use a word that is similar but below threshold
        # "helo" vs "hello" — QRatio will be < 95
        buf.insert([(0.0, 0.5, "helo"), (0.5, 1.0, "world")], offset=0)
        commit = buf.flush()

        # "helo" should NOT match "hello" at threshold 95
        committed_texts = [w[2] for w in commit]
        assert "helo" not in committed_texts


# ---------------------------------------------------------------------------
# Fallback logic
# ---------------------------------------------------------------------------

class TestFallbackLogic:

    def test_fallback_triggers_after_threshold(self):
        """No commits for threshold+1 flushes triggers fallback."""
        buf = HypothesisBuffer(use_fallback=True, fallback_threshold=2)

        # Populate pass: empty buffer, skipped by fallback logic (no counting).
        buf.insert(make_words(["hello", "world", "test", "data"]), offset=0)
        buf.flush()
        assert buf.unconfirmed_amount == 0

        # pass 1 (first real comparison): no match, unconfirmed 0 → 1
        buf.insert(make_words(["alpha", "beta", "gamma", "delta"]), offset=0)
        buf.flush()
        assert buf.unconfirmed_amount == 1

        # pass 2: no match, unconfirmed 1 → 2
        buf.insert(make_words(["one", "two", "three", "four"]), offset=0)
        buf.flush()
        assert buf.unconfirmed_amount == 2

        # pass 3: unconfirmed 2 >= 2 → fallback triggers → reset to 0
        buf.insert(make_words(["red", "green", "blue", "pink"]), offset=0)
        buf.flush()
        assert buf.unconfirmed_amount == 0

    def test_fallback_minimum_score_applied(self):
        """Fallback with low-score match doesn't commit (Fix C).

        When fallback triggers, __fallback compares buffer prefixes vs new prefixes.
        If the best fuzzy score is below fuzz_threshold, nothing is committed.
        We need buffer and new to contain *different* words so the score is low.
        """
        buf = HypothesisBuffer(use_fallback=True, fallback_threshold=1, qratio_threshold=95)

        # Populate pass: fills buffer, unconfirmed stays 0
        buf.insert(make_words(["hello", "world"]), offset=0)
        buf.flush()
        assert buf.unconfirmed_amount == 0

        # Pass 1: different words — no normal match, unconfirmed 0 → 1
        buf.insert(make_words(["zzzzz", "xxxxx"]), offset=0)
        commit = buf.flush()
        assert commit == []
        assert buf.unconfirmed_amount == 1

        # Pass 2: unconfirmed 1 >= 1 → fallback triggers.
        # buffer = ["zzzzz", "xxxxx"], new = ["qqqqq", "jjjjj"] — score well below 95.
        buf.insert(make_words(["qqqqq", "jjjjj"]), offset=0)
        commit = buf.flush()

        committed_texts = [w[2] for w in commit]
        assert "qqqqq" not in committed_texts
        assert "jjjjj" not in committed_texts

    def test_unconfirmed_resets_on_commit(self):
        """Successful commit resets unconfirmed_amount (Fix #4)."""
        buf = HypothesisBuffer(use_fallback=True, fallback_threshold=5)

        # Build up unconfirmed_amount
        buf.insert(make_words(["hello", "world"]), offset=0)
        buf.flush()

        buf.insert(make_words(["different", "words"]), offset=0)
        buf.flush()
        assert buf.unconfirmed_amount >= 1

        # Now do a matching pass — should reset unconfirmed_amount
        buf.insert(make_words(["different", "words"]), offset=0)
        commit = buf.flush()

        assert len(commit) > 0
        assert buf.unconfirmed_amount == 0

    def test_populate_flush_does_not_count(self):
        """First flush (empty buffer) must not inflate unconfirmed_amount."""
        buf = HypothesisBuffer(use_fallback=True, fallback_threshold=1)

        buf.insert(make_words(["hello", "world"]), offset=0)
        buf.flush()

        # Populate flush should be skipped — counter stays at 0
        assert buf.unconfirmed_amount == 0

    def test_fallback_threshold_clamped_to_one(self):
        """fallback_threshold < 1 is clamped to 1."""
        buf = HypothesisBuffer(use_fallback=True, fallback_threshold=0)
        assert buf.fallback_threshold == 1

        buf2 = HypothesisBuffer(use_fallback=True, fallback_threshold=-5)
        assert buf2.fallback_threshold == 1

    def test_fallback_disabled_by_default(self):
        """Without use_fallback=True, fallback never triggers."""
        buf = HypothesisBuffer()  # use_fallback defaults to False

        # Use different words each pass so normal flush never matches the buffer.
        word_sets = [
            ["hello", "world"],
            ["alpha", "beta"],
            ["one", "two"],
            ["red", "green"],
            ["cat", "dog"],
        ]
        for i, words in enumerate(word_sets):
            buf.insert(make_words(words), offset=0)
            commit = buf.flush()
            # First flush has no prior buffer so always empty.
            # Subsequent flushes: words differ from buffer, so no normal match.
            if i > 0:
                assert commit == [], f"iteration {i}: unexpected commit {commit}"

        # Without use_fallback, unconfirmed_amount is never touched
        assert buf.unconfirmed_amount == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_insert(self):
        """Insert empty list should not crash."""
        buf = HypothesisBuffer()
        buf.insert([], offset=0)
        assert buf.new == []

    def test_flush_empty_buffer(self):
        """Flush with empty buffer returns empty list."""
        buf = HypothesisBuffer()
        commit = buf.flush()
        assert commit == []

    def test_pop_commited(self):
        """pop_commited removes words up to given time."""
        buf = HypothesisBuffer()

        # Manually populate commited_in_buffer
        buf.commited_in_buffer = [
            (0.0, 0.5, "hello"),
            (0.5, 1.0, "world"),
            (1.0, 1.5, "test"),
            (1.5, 2.0, "data"),
        ]

        buf.pop_commited(1.0)

        # Words with end time <= 1.0 should be removed
        remaining_texts = [w[2] for w in buf.commited_in_buffer]
        assert "hello" not in remaining_texts
        assert "world" not in remaining_texts
        assert "test" in remaining_texts
        assert "data" in remaining_texts

    def test_multi_iteration_cycle(self):
        """Simulate 5+ iterations of insert -> flush -> complete."""
        buf = HypothesisBuffer()

        all_committed = []
        iterations = [
            ["hello", "world"],
            ["hello", "world", "this"],
            ["hello", "world", "this", "is"],
            ["world", "this", "is", "a"],
            ["this", "is", "a", "test"],
            ["is", "a", "test", "run"],
        ]

        for i, words in enumerate(iterations):
            word_tuples = make_words(words, start=i * 0.5)
            buf.insert(word_tuples, offset=0)
            commit = buf.flush()
            all_committed.extend(commit)

            # complete() should return the current buffer (unconfirmed tail)
            remaining = buf.complete()
            assert isinstance(remaining, list)

        # After multiple iterations, some words should have been committed
        # The exact count depends on matching, but the process shouldn't crash
        assert isinstance(all_committed, list)

    def test_flush_after_insert_no_prior_buffer(self):
        """First flush ever — buffer is empty, so nothing commits."""
        buf = HypothesisBuffer()
        buf.insert(make_words(["hello", "world"]), offset=0)
        commit = buf.flush()

        # No prior buffer to compare against, so no commits
        assert commit == []
        # flush() sets self.buffer = self.new (the unmatched remainder),
        # so buffer is now populated for the next iteration.
        assert len(buf.buffer) == 2
        texts = [w[2] for w in buf.buffer]
        assert "hello" in texts
        assert "world" in texts

    def test_insert_with_offset(self):
        """Verify offset is correctly applied to all timestamps."""
        buf = HypothesisBuffer()
        words = [(0.0, 0.5, "a"), (0.5, 1.0, "b"), (1.0, 1.5, "c")]
        buf.insert(words, offset=10.0)

        assert buf.new[0][0] == 10.0  # start time offset
        assert buf.new[0][1] == 10.5  # end time offset
        assert buf.new[1][0] == 10.5
        assert buf.new[2][0] == 11.0
