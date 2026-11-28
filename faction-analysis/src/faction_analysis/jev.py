# pattern: Imperative Shell
"""TypeSafe Jev labelling behind a small protocol.

Jev never runs at scale: it labels a bounded teaching sample per refresh. One request per pair, with the
reply next to its parent, because the model loses accuracy when a state carries unrelated text. Every attempt
counts against the request and token budget. A pair whose retries are exhausted stays unlabelled in this
refresh and falls through to the weak labels.
"""

import asyncio
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import httpx2
from typesafe_sdk import (
    AsyncTypeSafeClient,
    RetryPolicy,
    Score,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeAuthenticationError,
    TypeSafeError,
)

from faction_analysis.config import LabelConfig
from faction_analysis.labels import QUOTE, REPLY

QUESTION = 'hostility'
LEVELS = ('neutral or friendly', 'dismissive or mocking', 'hostile', 'threatening')
HOSTILE_LEVEL = 2
THREATENING_LEVEL = 3
INSTRUCTIONS = (
    'The state holds a post and the post it answers: a reply with its parent, or a quote with the quoted '
    'post. Score how the author of the first text treats the author of the second. Judge the tone and the '
    'intent toward that person, not the topic. Contempt and sarcasm are mocking. Insults, slurs and wishes '
    'of harm are hostile. Threats of violence, doxxing or harassment are threatening.'
)
# Bytes of the question and of the JSON keys around the two texts. A token is at least one byte, so the
# bytes of a request are an upper bound on its input tokens.
_QUESTION_BYTES = len(INSTRUCTIONS) + sum(len(level) for level in LEVELS) + 96
_MAX_UTF8_BYTES_PER_CHAR = 4


@dataclass(frozen=True)
class TextPair:
    """A reply with its parent, or a quote with the quoted post."""

    text: str
    parent: str
    kind: int

    def truncated(self, max_chars: int) -> TextPair:
        return TextPair(self.text[:max_chars], self.parent[:max_chars], self.kind)

    def state(self) -> dict[str, str]:
        if self.kind == QUOTE:
            return {'quote': self.text, 'quoted': self.parent}
        return {'reply': self.text, 'parent': self.parent}

    def token_bound(self) -> int:
        return _MAX_UTF8_BYTES_PER_CHAR * (len(self.text) + len(self.parent)) + _QUESTION_BYTES


@dataclass(frozen=True)
class JevLabel:
    """One answer. ``h`` is the probability of hostile or threatening."""

    h: float
    p_hostile: float
    p_threat: float
    model: str
    input_tokens: int


class JevLabeller(Protocol):
    async def label(self, pair: TextPair) -> JevLabel: ...


class RequestBudget:
    """Hard limits on requests and input tokens for one refresh.

    A request reserves its worst-case token count before it is sent and replaces it with the real count
    afterwards. Concurrent requests therefore cannot jointly spend more than the limit.
    """

    def __init__(self, *, max_requests: int, max_tokens: int) -> None:
        if max_requests < 0 or max_tokens < 0:
            raise ValueError('budget limits must not be negative')
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self.requests = 0
        self.reserved_tokens = 0
        self.tokens = 0

    def allows(self, token_bound: int) -> bool:
        return self.requests < self.max_requests and self.reserved_tokens + token_bound <= self.max_tokens

    def reserve(self, token_bound: int) -> None:
        self.requests += 1
        self.reserved_tokens += token_bound

    def charge(self, token_bound: int, input_tokens: int) -> None:
        self.reserved_tokens += input_tokens - token_bound
        self.tokens += input_tokens


def per_request_bound(config: LabelConfig) -> int:
    """The largest token count one request can have under the text limit."""
    return TextPair('x' * config.jev_max_text_chars, 'x' * config.jev_max_text_chars, REPLY).token_bound()


def new_budget(config: LabelConfig) -> RequestBudget:
    return RequestBudget(max_requests=config.jev_max_requests, max_tokens=config.jev_max_input_tokens)


class TypeSafeJevLabeller:
    """``AsyncTypeSafeClient`` with the fixed question and the pinned model.

    The SDK's own retries are off unless a policy is passed: they would send requests that the budget of the
    refresh never sees. ``label_pairs`` retries instead, and counts every attempt.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        max_text_chars: int,
        transport: httpx2.AsyncBaseTransport | None = None,
        compression: Literal['zstd', False] | None = None,
        retry: RetryPolicy | None = None,
    ) -> None:
        if not api_key:
            raise ValueError('a TypeSafe API key is required')
        self._model = model
        self._max_text_chars = max_text_chars
        self._client = AsyncTypeSafeClient(
            api_key=api_key,
            model=model,
            transport=transport,
            compression=compression,
            retry=retry if retry is not None else RetryPolicy(max_retries=0),
        )

    async def label(self, pair: TextPair) -> JevLabel:
        response = await self._client.system_one(
            state=pair.truncated(self._max_text_chars).state(),
            questions={QUESTION: Score(instructions=INSTRUCTIONS, criteria=list(LEVELS))},
            model=self._model,
        )
        answer = response.scores[QUESTION]
        if set(answer.probabilities) - set(range(len(LEVELS))):
            raise ValueError('Jev returned a level outside the rubric')
        p_hostile = float(answer.probabilities.get(HOSTILE_LEVEL, 0.0))
        p_threat = float(answer.probabilities.get(THREATENING_LEVEL, 0.0))
        h = p_hostile + p_threat
        if not math.isfinite(h) or not 0.0 <= h <= 1.0:
            raise ValueError('Jev returned a probability outside [0, 1]')
        return JevLabel(
            h=h,
            p_hostile=p_hostile,
            p_threat=p_threat,
            model=response.model,
            input_tokens=int(response.usage.input_tokens or 0),
        )

    async def close(self) -> None:
        await self._client.aclose()


LabelStore = Callable[[list[tuple[int, JevLabel]]], None]

ATTEMPTS_PER_PAIR = 3
"""How often a pair is sent before it stays unlabelled in this refresh: the first try and two retries."""
RETRY_STATUSES = frozenset({408, 429})
BACKOFF_SECONDS = 0.5
MAX_BACKOFF_SECONDS = 30.0
STORE_BATCH = 500
"""Answers are stored each time this many have arrived, and again when the labelling ends."""


def transient(exc: BaseException) -> bool:
    """True for a failure that another attempt may cure.

    No response or a timeout, a rate limit, a server error, or an answer that could not be read. A request
    the server rejects is final for this refresh.
    """
    if isinstance(exc, TypeSafeAPIConnectionError | TypeSafeAPIResponseValidationError):
        return True
    if isinstance(exc, TypeSafeAPIError):
        return exc.status in RETRY_STATUSES or exc.status >= 500
    # An answer outside the rubric or a local I/O error.
    return isinstance(exc, ValueError | OSError | asyncio.TimeoutError)


def backoff(exc: BaseException, attempt: int) -> float:
    """Seconds to wait before the next attempt: what the server asked for, or a doubling wait."""
    wished = getattr(exc, 'retry_after_ms', None)
    if isinstance(wished, int | float):
        return min(max(wished / 1000, 0.0), MAX_BACKOFF_SECONDS)
    return min(BACKOFF_SECONDS * 2**attempt, MAX_BACKOFF_SECONDS)


async def label_pairs(
    labeller: JevLabeller,
    pairs: Sequence[tuple[int, TextPair]],
    budget: RequestBudget,
    *,
    concurrency: int,
    max_text_chars: int,
    store: LabelStore | None = None,
    store_batch: int = STORE_BATCH,
) -> dict[int, JevLabel]:
    """Label every pair the budget still allows. A pair that fails stays out of the result.

    Each pair is cut to ``max_text_chars`` per text first, so its token bound is the bound of what is sent.
    Every attempt reserves a request and that bound, so neither limit can be exceeded however often a pair
    is tried, and a pair is tried at most ``ATTEMPTS_PER_PAIR`` times. At most ``concurrency`` requests are
    in flight.

    An authentication error stops the whole refresh: the pairs that have not been sent yet are left alone,
    and the error is raised once no request is on its way. Any other error keeps the reservation of that
    attempt spent, because a request that failed may still have been billed. A failure another attempt may
    cure is retried after a short wait. The next refresh revisits a pair that has no label stored.

    ``store`` receives the answers in batches, as soon as they exist, so that a crash or a later error cannot
    lose a label that was paid for. It runs on the event loop and must be quick. When it fails, no further
    pair is sent and the error is raised: labels must not be bought faster than they can be kept. A failure
    can cost the answers that are not stored yet: at most one batch and the requests in flight.
    """
    if concurrency < 1:
        raise ValueError('concurrency must be positive')
    semaphore = asyncio.Semaphore(concurrency)
    results: dict[int, JevLabel] = {}
    pending: list[tuple[int, JevLabel]] = []
    # The first failure that ends the refresh: the key stopped working, or the store cannot take answers.
    # Every pair that has not been sent yet is then left alone.
    fatal: list[BaseException] = []

    def flush() -> None:
        """Hand the pending answers to the store. They stay pending when the store fails."""
        if store is not None and pending:
            batch = list(pending)
            store(batch)
            pending.clear()

    async def one(uri_hash: int, pair: TextPair) -> None:
        pair = pair.truncated(max_text_chars)
        bound = pair.token_bound()
        async with semaphore:
            for attempt in range(ATTEMPTS_PER_PAIR):
                if fatal or not budget.allows(bound):
                    return
                budget.reserve(bound)
                try:
                    label = await labeller.label(pair)
                except TypeSafeAuthenticationError as exc:
                    fatal.append(exc)
                    return
                except (TypeSafeError, ValueError, OSError, asyncio.TimeoutError) as exc:
                    budget.charge(bound, bound)
                    if not transient(exc) or attempt == ATTEMPTS_PER_PAIR - 1:
                        return
                    await asyncio.sleep(backoff(exc, attempt))
                    continue
                budget.charge(bound, label.input_tokens)
                results[uri_hash] = label
                pending.append((uri_hash, label))
                if len(pending) >= store_batch:
                    try:
                        flush()
                    except Exception as exc:  # noqa: BLE001 - no label is bought after the store failed
                        fatal.append(exc)
                        return
                return

    try:
        await asyncio.gather(*(one(uri_hash, pair) for uri_hash, pair in pairs))
    finally:
        # The answers that were paid for reach the store whatever happens to the rest.
        try:
            flush()
        except Exception as flush_error:
            if not fatal:
                raise
            # The first failure stays the error. The second is attached, not put in its place.
            fatal[0].add_note(f'the last {len(pending)} answers could not be stored: {flush_error!r}')
    # Raised after every task has settled, so that no request is still on its way when the caller acts.
    if fatal:
        raise fatal[0]
    return results


def label_pairs_sync(
    labeller: JevLabeller,
    pairs: Sequence[tuple[int, TextPair]],
    budget: RequestBudget,
    *,
    concurrency: int,
    max_text_chars: int,
    store: LabelStore | None = None,
) -> dict[int, JevLabel]:
    return asyncio.run(
        label_pairs(labeller, pairs, budget, concurrency=concurrency, max_text_chars=max_text_chars, store=store)
    )
