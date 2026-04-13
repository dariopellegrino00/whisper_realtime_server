import logging
import sys

from rapidfuzz import fuzz, utils

logger = logging.getLogger(__name__)

MAX_DEDUP_NGRAM_SIZE = 10
DEDUP_MAX_FORWARD_GAP_SECONDS = 0.2


class HypothesisBuffer:
    def __init__(self, logfile=sys.stderr, **kwargs):
        self.commited_in_buffer = []
        self.buffer = []
        self.new = []
        self.last_commited_time = 0
        self.last_commited_word = None
        self.fuzz_threshold = kwargs.get("qratio_threshold", 95)
        self.dedup_threshold = kwargs.get("dedup_threshold", 98)
        self.use_fallback = kwargs.get("use_fallback", True)
        fallback_threshold = kwargs.get("fallback_threshold", 1)
        if fallback_threshold < 1:
            logger.warning(
                "fallback_threshold must be >= 1, got %s — clamping to 1",
                fallback_threshold,
            )
        self.fallback_threshold = max(1, fallback_threshold)
        self.unconfirmed_amount = 0
        self.logfile = logfile

    def insert(self, new, offset):
        self.new = [
            (start + offset, end + offset, text)
            for start, end, text in new
            if start + offset > self.last_commited_time - 0.1
        ]

        if len(self.new) < 1:
            return

        start, _end, _text = self.new[0]
        if not self.commited_in_buffer:
            return

        # Dedup should only handle decode overlap near the last committed boundary.
        # A larger forward gap is more likely to be a real repetition than overlap.
        if start > self.last_commited_time + DEDUP_MAX_FORWARD_GAP_SECONDS:
            return

        if self.last_commited_time - start >= 1:
            return

        committed_length = len(self.commited_in_buffer)
        new_length = len(self.new)
        for ngram_size in range(min(committed_length, new_length, MAX_DEDUP_NGRAM_SIZE), 0, -1):
            committed_tail = " ".join(word[2] for word in self.commited_in_buffer[-ngram_size:])
            new_head = " ".join(word[2] for word in self.new[:ngram_size])
            if (
                fuzz.QRatio(
                    committed_tail,
                    new_head,
                    processor=utils.default_process,
                )
                >= self.dedup_threshold
            ):
                for _ in range(ngram_size):
                    self.new.pop(0)
                logger.debug("dedup: removed %d overlapping word(s)", ngram_size)
                break

    def __commit_and_pop(self, num_new_pops, num_buffer_pops, commit):
        if num_new_pops <= 0:
            return

        committed_words = self.new[:num_new_pops]
        for word in committed_words:
            commit.append((word[0], word[1], word[2]))

        self.last_commited_word = committed_words[-1][2]
        self.last_commited_time = committed_words[-1][1]
        self.new = self.new[num_new_pops:]
        self.buffer = self.buffer[num_buffer_pops:]

    def __fallback(self, commit):
        if not self.buffer or not self.new:
            return

        half = len(self.buffer) // 2
        prefixes = [
            " ".join(word[-1] for word in self.buffer[: index + 1]) for index in range(half + 1)
        ]

        max_score = 0
        half_time = self.buffer[half][1] + 1
        new_filtered = [entry for entry in self.new if entry[1] <= half_time]
        num_drops_buffer = 0
        num_drops_new = 0

        for new_index in range(len(new_filtered)):
            new_prefix = " ".join(word[-1] for word in new_filtered[: new_index + 1])
            for buffer_index, prefix in enumerate(prefixes):
                score = fuzz.QRatio(prefix, new_prefix)
                if score > max_score:
                    max_score = score
                    num_drops_buffer = buffer_index + 1
                    num_drops_new = new_index + 1

        self.__commit_and_pop(num_drops_new, num_drops_buffer, commit)

    def flush(self):
        commit = []
        while self.new:
            if len(self.buffer) == 0:
                break

            if (
                fuzz.QRatio(
                    self.new[0][2],
                    self.buffer[0][2],
                    processor=utils.default_process,
                )
                >= self.fuzz_threshold
            ):
                self.__commit_and_pop(1, 1, commit)
            else:
                break

        if self.use_fallback:
            if commit:
                self.unconfirmed_amount = 0
            elif not self.buffer:
                pass
            elif self.unconfirmed_amount >= self.fallback_threshold:
                self.__fallback(commit)
                self.unconfirmed_amount = 0
            else:
                self.unconfirmed_amount += 1

        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time):
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        return self.buffer
