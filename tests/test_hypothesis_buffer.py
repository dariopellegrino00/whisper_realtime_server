"""Unit tests for the HypothesisBuffer in whisper_online.py."""

import sys
import os

# Ensure the src directory is in the path so we can import whisper_online
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from whisper_online import HypothesisBuffer


def make_words(texts, start=0.0, step=0.5):
    """Helper to quickly create a list of (start, end, text) word tuples."""
    words = []
    t = start
    for w in texts:
        words.append((t, t + step, w))
        t += step
    return words


# Basic lifecycle and word insertion tests

class TestBasicLifecycle:

    def test_insert_basic(self):
        """Verify that words are correctly inserted and the offset is applied."""
        buf = HypothesisBuffer()
        words = [(0.0, 0.5, "hello"), (0.5, 1.0, "world")]
        buf.insert(words, offset=1.0)
        # The words in 'new' should now have their timestamps shifted by the offset
        assert len(buf.new) == 2
        assert buf.new[0] == (1.0, 1.5, "hello")
        assert buf.new[1] == (1.5, 2.0, "world")

    def test_flush_confirms_matching_words(self):
        """Simulate two iterations where the same words are seen, triggering a confirmation."""
        buf = HypothesisBuffer()
        words = [(0.0, 0.5, "hello"), (0.5, 1.0, "world")]

        # First pass: fill the buffer
        buf.insert(words, offset=0)
        buf.flush()

        # Second pass: identical words should now be confirmed
        buf.insert(words, offset=0)
        commit = buf.flush()

        committed_texts = [w[2] for w in commit]
        assert "hello" in committed_texts
        assert "world" in committed_texts

    def test_flush_no_match_no_commit(self):
        """Ensure different words across iterations do not trigger a confirmation."""
        buf = HypothesisBuffer()

        # First pass
        buf.insert([(0.0, 0.5, "hello"), (0.5, 1.0, "world")], offset=0)
        buf.flush()

        # Second pass: completely different words shouldn't match anything
        buf.insert([(0.0, 0.5, "foo"), (0.5, 1.0, "bar")], offset=0)
        commit = buf.flush()

        assert commit == []

    def test_complete_returns_buffer(self):
        """Verify that complete() returns the currently unconfirmed words."""
        buf = HypothesisBuffer()
        words = [(0.0, 0.5, "hello"), (0.5, 1.0, "world")]

        buf.insert(words, offset=0)
        buf.flush()

        result = buf.complete()
        texts = [w[2] for w in result]
        assert "hello" in texts
        assert "world" in texts


# Tests for the deduplication logic during insertion

class TestInsertDedup:

    def test_insert_dedup_removes_overlap(self):
        """Verify that newly inserted words that overlap with already committed ones are removed."""
        buf = HypothesisBuffer()

        # Commit some initial words
        buf.insert([(0.0, 0.5, "hello"), (0.5, 1.0, "world")], offset=0)
        buf.flush()
        buf.insert([(0.0, 0.5, "hello"), (0.5, 1.0, "world")], offset=0)
        buf.flush()

        # Insert words that overlap at the boundary
        buf.insert([(0.9, 1.4, "world"), (1.4, 1.9, "test")], offset=0)

        new_texts = [w[2] for w in buf.new]
        # "world" is already committed and should be deduped
        assert "test" in new_texts
        assert len(buf.new) <= 1

    def test_insert_dedup_ngram_sizes(self):
        """Test deduplication for multiple words (n-grams)."""
        buf = HypothesisBuffer()

        # Set up a committed buffer with a 3-word tail
        words = [(0.0, 0.5, "one"), (0.5, 1.0, "two"), (1.0, 1.5, "three")]
        buf.insert(words, offset=0)
        buf.flush()
        buf.insert(words, offset=0)
        buf.flush()

        # New sequence starting with a 2-word overlap
        new_words = [(0.5, 1.0, "two"), (1.0, 1.5, "three"), (1.5, 2.0, "four")]
        buf.insert(new_words, offset=0)

        new_texts = [w[2] for w in buf.new]
        assert "four" in new_texts
        assert "two" not in new_texts

    def test_insert_dedup_repeated_words(self):
        """Ensure dedup finds the longest possible overlap when words repeat."""
        buf = HypothesisBuffer()

        # Committed: "go home go home"
        words = [(0.0, 0.5, "go"), (0.5, 1.0, "home"), (1.0, 1.5, "go"), (1.5, 2.0, "home")]
        buf.insert(words, offset=0)
        buf.flush()
        buf.insert(words, offset=0)
        buf.flush()

        # New: "home go home now"
        new_words = [(1.0, 1.5, "home"), (1.5, 2.0, "go"), (2.0, 2.5, "home"), (2.5, 3.0, "now")]
        buf.insert(new_words, offset=0)

        new_texts = [w[2] for w in buf.new]
        assert new_texts == ["now"]

    def test_insert_dedup_custom_threshold(self):
        """Verify that the similarity threshold for deduplication is configurable."""
        # A 100% threshold means even a tiny difference prevents deduplication
        buf = HypothesisBuffer(dedup_threshold=100)

        words = [(0.0, 0.5, "hello"), (0.5, 1.0, "world")]
        buf.insert(words, offset=0)
        buf.flush()
        buf.insert(words, offset=0)
        buf.flush()

        # "wrld" is not an exact match for "world"
        buf.insert([(1.0, 1.5, "wrld"), (1.5, 2.0, "test")], offset=0)
        new_texts = [w[2] for w in buf.new]
        assert "wrld" in new_texts

    def test_insert_filters_old_words(self):
        """Old words with timestamps significantly before the last commit should be ignored."""
        buf = HypothesisBuffer()
        buf.last_commited_time = 5.0

        words = [(1.0, 2.0, "old"), (4.0, 4.5, "also_old"), (5.0, 5.5, "new")]
        buf.insert(words, offset=0)

        new_texts = [w[2] for w in buf.new]
        assert "old" not in new_texts
        assert "also_old" not in new_texts
        assert "new" in new_texts

    def test_insert_no_dedup_when_far_apart(self):
        """Skip deduplication if the new audio is far ahead of the last commit."""
        buf = HypothesisBuffer()
        buf.last_commited_time = 1.0
        buf.last_commited_word = "hello"
        buf.commited_in_buffer = [(0.5, 1.0, "hello")]

        # Audio starting at 5.0 is more than 1 second away from 1.0
        words = [(5.0, 5.5, "hello"), (5.5, 6.0, "world")]
        buf.insert(words, offset=0)

        new_texts = [w[2] for w in buf.new]
        assert "hello" in new_texts
        assert "world" in new_texts


# Tests for the word confirmation (flush) logic

class TestFlushConfirmation:

    def test_flush_partial_match(self):
        """Only confirm the prefix that actually matches the previous iteration."""
        buf = HypothesisBuffer()

        # Iteration 1
        buf.insert([(0.0, 0.5, "hello"), (0.5, 1.0, "world"), (1.0, 1.5, "foo")], offset=0)
        buf.flush()

        # Iteration 2: change the third word
        buf.insert([(0.0, 0.5, "hello"), (0.5, 1.0, "world"), (1.0, 1.5, "bar")], offset=0)
        commit = buf.flush()

        committed_texts = [w[2] for w in commit]
        assert "hello" in committed_texts
        assert "world" in committed_texts
        assert "bar" not in committed_texts

    def test_flush_threshold_boundary(self):
        """Fuzzy matching should respect the qratio_threshold."""
        buf = HypothesisBuffer(qratio_threshold=95)

        # Iteration 1
        buf.insert([(0.0, 0.5, "hello"), (0.5, 1.0, "world")], offset=0)
        buf.flush()

        # Iteration 2: "helo" is slightly different from "hello"
        buf.insert([(0.0, 0.5, "helo"), (0.5, 1.0, "world")], offset=0)
        commit = buf.flush()

        committed_texts = [w[2] for w in commit]
        assert "helo" not in committed_texts


# Tests for the fallback mechanism when no matches occur for a while

class TestFallbackLogic:

    def test_fallback_triggers_after_threshold(self):
        """Verify that fallback triggers after the configured number of mismatches."""
        buf = HypothesisBuffer(use_fallback=True, fallback_threshold=2)

        # Initial pass just fills the buffer
        buf.insert(make_words(["hello", "world", "test", "data"]), offset=0)
        buf.flush()

        # Pass 1: mismatch
        buf.insert(make_words(["alpha", "beta", "gamma", "delta"]), offset=0)
        buf.flush()
        assert buf.unconfirmed_amount == 1

        # Pass 2: mismatch
        buf.insert(make_words(["one", "two", "three", "four"]), offset=0)
        buf.flush()
        assert buf.unconfirmed_amount == 2

        # Pass 3: triggers fallback because threshold (2) is reached
        buf.insert(make_words(["red", "green", "blue", "pink"]), offset=0)
        buf.flush()
        assert buf.unconfirmed_amount == 0

    def test_fallback_commits_best_prefix_even_with_low_score(self):
        """Fallback should be more permissive and commit the best available prefix."""
        buf = HypothesisBuffer(use_fallback=True, fallback_threshold=1, qratio_threshold=95)

        buf.insert(make_words(["hello", "world"]), offset=0)
        buf.flush()

        # Mismatch pass
        buf.insert(make_words(["zzzzz", "xxxxx"]), offset=0)
        buf.flush()
        assert buf.unconfirmed_amount == 1

        # Triggers fallback
        buf.insert(make_words(["qqqqq", "jjjjj"]), offset=0)
        commit = buf.flush()

        committed_texts = [w[2] for w in commit]
        assert "qqqqq" in committed_texts
        assert "jjjjj" in committed_texts

    def test_fallback_ignores_words_beyond_filtered_window(self):
        """The fallback logic should respect time boundaries when comparing prefixes."""
        buf = HypothesisBuffer(use_fallback=True, fallback_threshold=1, qratio_threshold=95)

        buf.insert(make_words(["alpha", "beta"], start=0.0, step=0.5), offset=0)
        buf.flush()

        buf.insert(make_words(["noise", "words"], start=0.0, step=0.5), offset=0)
        buf.flush()

        # Create a state where one word is in the window and the other is way ahead
        buf.buffer = make_words(["alpha", "beta"], start=0.0, step=0.5)
        buf.new = [
            (0.0, 0.5, "alpha"),
            (3.0, 3.5, "beta"),
        ]

        commit = []
        buf._HypothesisBuffer__fallback(commit)

        committed_texts = [w[2] for w in commit]
        assert committed_texts == ["alpha"]

    def test_unconfirmed_resets_on_commit(self):
        """Any successful normal commit should reset the unconfirmed counter."""
        buf = HypothesisBuffer(use_fallback=True, fallback_threshold=5)

        buf.insert(make_words(["hello", "world"]), offset=0)
        buf.flush()

        buf.insert(make_words(["different", "words"]), offset=0)
        buf.flush()
        assert buf.unconfirmed_amount >= 1

        # A normal matching pass should clear the counter
        buf.insert(make_words(["different", "words"]), offset=0)
        commit = buf.flush()

        assert len(commit) > 0
        assert buf.unconfirmed_amount == 0

    def test_populate_flush_does_not_count(self):
        """The very first flush shouldn't count as a mismatch since there was no prior buffer."""
        buf = HypothesisBuffer(use_fallback=True, fallback_threshold=1)

        buf.insert(make_words(["hello", "world"]), offset=0)
        buf.flush()

        assert buf.unconfirmed_amount == 0

    def test_fallback_threshold_clamped_to_one(self):
        """Negative or zero thresholds for fallback should be automatically set to 1."""
        buf = HypothesisBuffer(use_fallback=True, fallback_threshold=0)
        assert buf.fallback_threshold == 1

        buf2 = HypothesisBuffer(use_fallback=True, fallback_threshold=-5)
        assert buf2.fallback_threshold == 1

    def test_fallback_disabled_by_default(self):
        """Ensure that the fallback mechanism is inactive unless explicitly enabled."""
        buf = HypothesisBuffer()  # Defaults to use_fallback=False

        word_sets = [
            ["hello", "world"],
            ["alpha", "beta"],
            ["one", "two"],
            ["red", "green"],
            ["cat", "dog"],
        ]
        for i, words in enumerate(word_sets):
            buf.insert(make_words(words), offset=0)
            buf.flush()

        assert buf.unconfirmed_amount == 0


# Edge cases and miscellaneous tests

class TestEdgeCases:

    def test_empty_insert(self):
        """Verify that inserting an empty list is handled gracefully."""
        buf = HypothesisBuffer()
        buf.insert([], offset=0)
        assert buf.new == []

    def test_flush_empty_buffer(self):
        """Flushing an empty buffer should just return an empty list."""
        buf = HypothesisBuffer()
        commit = buf.flush()
        assert commit == []

    def test_pop_commited(self):
        """Verify that pop_commited correctly prunes the internal history."""
        buf = HypothesisBuffer()

        buf.commited_in_buffer = [
            (0.0, 0.5, "hello"),
            (0.5, 1.0, "world"),
            (1.0, 1.5, "test"),
            (1.5, 2.0, "data"),
        ]

        buf.pop_commited(1.0)

        remaining_texts = [w[2] for w in buf.commited_in_buffer]
        assert "hello" not in remaining_texts
        assert "world" not in remaining_texts
        assert "test" in remaining_texts
        assert "data" in remaining_texts

    def test_multi_iteration_cycle(self):
        """Test a long sequence of insertions and flushes to check for cumulative errors."""
        buf = HypothesisBuffer()

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
            buf.flush()
            buf.complete()

    def test_flush_after_insert_no_prior_buffer(self):
        """Verify the state after the first ever insertion and flush."""
        buf = HypothesisBuffer()
        buf.insert(make_words(["hello", "world"]), offset=0)
        commit = buf.flush()

        assert commit == []
        assert len(buf.buffer) == 2

    def test_insert_with_offset(self):
        """Verify that timestamps are correctly offset during insertion."""
        buf = HypothesisBuffer()
        words = [(0.0, 0.5, "a"), (0.5, 1.0, "b"), (1.0, 1.5, "c")]
        buf.insert(words, offset=10.0)

        assert buf.new[0][0] == 10.0
        assert buf.new[0][1] == 10.5
        assert buf.new[1][0] == 10.5
        assert buf.new[2][0] == 11.0
