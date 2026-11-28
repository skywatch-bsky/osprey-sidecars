import asyncio
import json

import httpx2
import pytest
from typesafe_sdk import (
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeAPITimeoutError,
    TypeSafeAuthenticationError,
    TypeSafeBadRequestError,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
)

from faction_analysis import jev
from faction_analysis.config import LabelConfig
from faction_analysis.labels import QUOTE, REPLY

MODEL = 'jev-1.14.0'


def answer(probabilities: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4), tokens: int = 120, model: str = MODEL) -> dict:
    return {
        'model': model,
        'usage': {'input_tokens': tokens, 'output_tokens': 0},
        'answers': {
            'hostility': {
                'type': 'score',
                'score': 2.0,
                'confidence': 0.9,
                'legend': {'0': 'a', '1': 'b', '2': 'c', '3': 'd'},
                'probabilities': {str(level): value for level, value in enumerate(probabilities)},
            }
        },
    }


class FakeTypeSafe:
    """Answers like the TypeSafe API. Records every request body. ``respond`` maps a request to its answer."""

    def __init__(self, respond) -> None:
        self.requests: list[dict] = []
        self._respond = respond

    def transport(self) -> httpx2.MockTransport:
        def handler(request: httpx2.Request) -> httpx2.Response:
            body = json.loads(request.content)
            self.requests.append(body)
            response = self._respond(len(self.requests), body)
            if isinstance(response, int):
                return httpx2.Response(response, json={'error': 'rejected'})
            return httpx2.Response(200, json=response)

        return httpx2.MockTransport(handler)

    def labeller(self, max_text_chars: int = 2000) -> jev.TypeSafeJevLabeller:
        # No retry policy: the labeller must switch the SDK's retries off by itself.
        return jev.TypeSafeJevLabeller(
            'key',
            MODEL,
            max_text_chars=max_text_chars,
            transport=self.transport(),
            compression=False,
        )


def pair(kind: int = REPLY, text: str = 'you are wrong', parent: str = 'I think otherwise') -> jev.TextPair:
    return jev.TextPair(text, parent, kind)


def label(labeller: jev.JevLabeller, text_pair: jev.TextPair) -> jev.JevLabel:
    return asyncio.run(labeller.label(text_pair))


def label_all(
    fake: FakeTypeSafe, pairs: list[tuple[int, jev.TextPair]], budget: jev.RequestBudget, *, concurrency: int = 4
) -> dict[int, jev.JevLabel]:
    return asyncio.run(jev.label_pairs(fake.labeller(), pairs, budget, concurrency=concurrency, max_text_chars=2000))


def same_length_pairs(count: int) -> list[tuple[int, jev.TextPair]]:
    """Pairs whose texts have the same length, so every request has the same token bound."""
    return [(index, pair(text='text', parent='parent')) for index in range(count)]


def test_fake_transport_labels_a_pair() -> None:
    fake = FakeTypeSafe(lambda index, body: answer())
    result = label(fake.labeller(), pair())
    assert result.h == pytest.approx(0.7)
    assert result.p_hostile == pytest.approx(0.3)
    assert result.p_threat == pytest.approx(0.4)
    assert (result.model, result.input_tokens) == (MODEL, 120)
    body = fake.requests[0]
    assert body['state'] == {'reply': 'you are wrong', 'parent': 'I think otherwise'}
    question = body['questions']['hostility']
    assert question['type'] == 'score'
    assert question['criteria'] == list(jev.LEVELS)
    assert question['instructions'] == jev.INSTRUCTIONS
    assert body['model'] == MODEL


def test_the_sdk_sends_no_request_of_its_own() -> None:
    """The SDK would retry a 500 twice on its own. Those requests would leave no trace in the budget."""
    fake = FakeTypeSafe(lambda index, body: 500)
    budget = jev.RequestBudget(max_requests=10, max_tokens=10**9)
    results = label_all(fake, same_length_pairs(1), budget)
    assert results == {}
    assert budget.requests == len(fake.requests) == jev.ATTEMPTS_PER_PAIR


def test_a_transient_failure_is_retried_inside_the_budget() -> None:
    # The first answer is a server error, the second is a rate limit, the third arrives.
    fake = FakeTypeSafe(lambda index, body: 500 if index == 1 else 429 if index == 2 else answer())
    budget = jev.RequestBudget(max_requests=10, max_tokens=10**9)
    results = label_all(fake, same_length_pairs(1), budget)
    assert list(results) == [0]
    assert budget.requests == len(fake.requests) == 3
    # The two failed attempts keep their whole bound as spent, the third is charged its real size.
    assert budget.tokens == 2 * pair(text='text', parent='parent').token_bound() + 120


def test_retries_stop_at_the_request_budget() -> None:
    fake = FakeTypeSafe(lambda index, body: 500)
    budget = jev.RequestBudget(max_requests=1, max_tokens=10**9)
    assert label_all(fake, same_length_pairs(1), budget) == {}
    assert len(fake.requests) == 1


def test_a_rejected_request_is_not_retried() -> None:
    fake = FakeTypeSafe(lambda index, body: 400)
    budget = jev.RequestBudget(max_requests=10, max_tokens=10**9)
    assert label_all(fake, same_length_pairs(1), budget) == {}
    assert len(fake.requests) == 1


def test_which_failures_another_attempt_may_cure() -> None:
    for exc in (
        TypeSafeAPIConnectionError('no route'),
        TypeSafeAPITimeoutError(1.0),
        TypeSafeRateLimitError(429, {}, httpx2.Headers()),
        TypeSafeInternalServerError(503, {}, httpx2.Headers()),
        TypeSafeAPIError(408, {}, httpx2.Headers()),
        TypeSafeAPIResponseValidationError(200, {}, httpx2.Headers(), 'answers'),
        ValueError('outside the rubric'),
    ):
        assert jev.transient(exc), exc
    for exc in (
        TypeSafeBadRequestError(400, {}, httpx2.Headers()),
        TypeSafeAuthenticationError(401, {}, httpx2.Headers()),
        TypeSafeAPIError(422, {}, httpx2.Headers()),
    ):
        assert not jev.transient(exc), exc
    # A rate limit that asks for a wait gets it, up to the cap. Anything else waits a doubling time.
    limited = TypeSafeRateLimitError(429, {}, httpx2.Headers({'retry-after-ms': '2500'}))
    assert jev.backoff(limited, 0) == pytest.approx(2.5)
    assert jev.backoff(TypeSafeRateLimitError(429, {}, httpx2.Headers()), 0) == jev.BACKOFF_SECONDS
    assert jev.backoff(ValueError('x'), 1) == 2 * jev.BACKOFF_SECONDS
    assert jev.backoff(ValueError('x'), 12) == jev.MAX_BACKOFF_SECONDS


def test_concurrency_is_held_through_the_request() -> None:
    class Counting:
        def __init__(self) -> None:
            self.active = 0
            self.peak = 0

        async def label(self, pair: jev.TextPair) -> jev.JevLabel:
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0.005)
            self.active -= 1
            return jev.JevLabel(0.5, 0.5, 0.0, MODEL, 10)

    counting = Counting()
    budget = jev.RequestBudget(max_requests=100, max_tokens=10**9)
    results = asyncio.run(jev.label_pairs(counting, same_length_pairs(20), budget, concurrency=2, max_text_chars=2000))
    assert len(results) == 20
    assert counting.peak == 2


def test_answers_are_stored_as_they_arrive() -> None:
    fake = FakeTypeSafe(lambda index, body: answer())
    batches: list[list[int]] = []
    budget = jev.RequestBudget(max_requests=100, max_tokens=10**9)
    results = asyncio.run(
        jev.label_pairs(
            fake.labeller(),
            same_length_pairs(10),
            budget,
            concurrency=4,
            max_text_chars=2000,
            store=lambda batch: batches.append([h for h, _ in batch]),
            store_batch=3,
        )
    )
    assert len(results) == 10
    assert [len(batch) for batch in batches] == [3, 3, 3, 1]
    assert sorted(hash for batch in batches for hash in batch) == list(range(10))


def test_paid_answers_survive_a_later_authentication_error() -> None:
    # The fifth request loses the key. The four answers before it are stored.
    fake = FakeTypeSafe(lambda index, body: 401 if index == 5 else answer())
    batches: list[list[int]] = []
    budget = jev.RequestBudget(max_requests=100, max_tokens=10**9)
    with pytest.raises(TypeSafeAuthenticationError):
        asyncio.run(
            jev.label_pairs(
                fake.labeller(),
                same_length_pairs(20),
                budget,
                concurrency=1,
                max_text_chars=2000,
                store=lambda batch: batches.append([h for h, _ in batch]),
                store_batch=3,
            )
        )
    stored = sorted(hash for batch in batches for hash in batch)
    assert len(stored) == 4 and len(batches[-1]) == 1


def test_a_failing_store_stops_the_labelling() -> None:
    """Labels must not be bought faster than they can be kept."""
    calls: list[int] = []
    attempts: list[list[int]] = []

    class Counting:
        async def label(self, pair: jev.TextPair) -> jev.JevLabel:
            calls.append(len(calls))
            await asyncio.sleep(0)
            return jev.JevLabel(0.5, 0.5, 0.0, MODEL, 10)

    def store(batch: list[tuple[int, jev.JevLabel]]) -> None:
        attempts.append([uri_hash for uri_hash, _ in batch])
        if len(attempts) >= 2:
            raise OSError('the cache is unreachable')

    budget = jev.RequestBudget(max_requests=100, max_tokens=10**9)
    with pytest.raises(OSError, match='unreachable'):
        asyncio.run(
            jev.label_pairs(
                Counting(),
                same_length_pairs(20),
                budget,
                concurrency=1,
                max_text_chars=2000,
                store=store,
                store_batch=1,
            )
        )
    # The first answer was stored and the second could not be. No third pair was sent.
    assert len(calls) == 2
    # The batch that could not be stored is tried again when the labelling ends, not dropped.
    assert attempts == [[0], [1], [1]]


def test_a_failing_store_does_not_hide_a_lost_key() -> None:
    fake = FakeTypeSafe(lambda index, body: 401 if index == 2 else answer())

    def store(batch: list[tuple[int, jev.JevLabel]]) -> None:
        raise OSError('the cache is unreachable')

    budget = jev.RequestBudget(max_requests=100, max_tokens=10**9)
    with pytest.raises(TypeSafeAuthenticationError) as lost:
        asyncio.run(
            jev.label_pairs(
                fake.labeller(),
                same_length_pairs(20),
                budget,
                concurrency=1,
                max_text_chars=2000,
                store=store,
                store_batch=2,
            )
        )
    # The lost key stays the error. The answer that could not be stored is named in a note.
    assert lost.value.__notes__ == ["the last 1 answers could not be stored: OSError('the cache is unreachable')"]
    assert len(fake.requests) == 2


def test_a_quote_is_sent_as_quote_and_quoted() -> None:
    fake = FakeTypeSafe(lambda index, body: answer())
    label(fake.labeller(), pair(QUOTE))
    assert fake.requests[0]['state'] == {'quote': 'you are wrong', 'quoted': 'I think otherwise'}


def test_texts_are_cut_to_the_limit() -> None:
    fake = FakeTypeSafe(lambda index, body: answer())
    long = 'x' * 500
    label(fake.labeller(max_text_chars=100), pair(text=long, parent=long))
    assert fake.requests[0]['state'] == {'reply': 'x' * 100, 'parent': 'x' * 100}


def test_h_is_the_probability_of_hostile_or_threatening() -> None:
    fake = FakeTypeSafe(lambda index, body: answer((0.05, 0.05, 0.0, 0.9)))
    assert label(fake.labeller(), pair()).h == pytest.approx(0.9)
    fake = FakeTypeSafe(lambda index, body: answer((0.25, 0.25, 0.25, 0.25)))
    assert label(fake.labeller(), pair()).h == pytest.approx(0.5)


def test_an_answer_outside_the_rubric_is_rejected() -> None:
    broken = answer()
    broken['answers']['hostility']['probabilities']['9'] = 0.5
    fake = FakeTypeSafe(lambda index, body: broken)
    with pytest.raises(ValueError, match='rubric'):
        label(fake.labeller(), pair())


def test_request_budget_is_hard() -> None:
    fake = FakeTypeSafe(lambda index, body: answer())
    budget = jev.RequestBudget(max_requests=3, max_tokens=10**9)
    results = label_all(fake, same_length_pairs(10), budget)
    assert len(results) == 3
    assert budget.requests == len(fake.requests) == 3


def test_token_budget_is_hard_even_with_concurrency() -> None:
    # Each answer reports almost the full bound. Two fit. A third would need the bound again.
    fake = FakeTypeSafe(lambda index, body: answer(tokens=560))
    bound = pair(text='text', parent='parent').token_bound()
    assert 560 <= bound < 2 * 560
    budget = jev.RequestBudget(max_requests=100, max_tokens=2 * bound + bound // 2)
    results = label_all(fake, same_length_pairs(20), budget, concurrency=8)
    assert len(results) == 2
    assert budget.requests == len(fake.requests) == 2
    assert budget.reserved_tokens <= budget.max_tokens
    assert budget.tokens == 1_120


def test_small_answers_spend_the_budget_on_their_real_size() -> None:
    # Five answers of 50 tokens are reported in full. The sixth is refused, because reserving the bound
    # for it would cross the limit although its real size would not.
    fake = FakeTypeSafe(lambda index, body: answer(tokens=50))
    bound = pair(text='text', parent='parent').token_bound()
    budget = jev.RequestBudget(max_requests=100, max_tokens=5 * 50 + bound - 1)
    results = label_all(fake, same_length_pairs(20), budget, concurrency=8)
    assert len(results) == 5
    assert budget.tokens == 250


def test_a_failed_pair_stays_unlabelled_and_keeps_its_reservation_spent() -> None:
    fake = FakeTypeSafe(lambda index, body: 400 if body['state']['reply'] == 'text 1' else answer())
    budget = jev.RequestBudget(max_requests=100, max_tokens=10**9)
    pairs = [(index, pair(text=f'text {index}', parent='parent')) for index in range(5)]
    results = label_all(fake, pairs, budget, concurrency=4)
    assert sorted(results) == [0, 2, 3, 4]
    assert budget.requests == 5
    # Four answers reported 120 tokens each. The failed request keeps its whole bound as spent, because it
    # may have been billed.
    failed_bound = pair(text='text 1', parent='parent').token_bound()
    assert budget.tokens == 4 * 120 + failed_bound
    assert budget.reserved_tokens == budget.tokens


def test_an_authentication_error_stops_the_refresh() -> None:
    fake = FakeTypeSafe(lambda index, body: 401)
    with pytest.raises(TypeSafeAuthenticationError):
        label_all(fake, same_length_pairs(5), jev.RequestBudget(max_requests=100, max_tokens=10**9), concurrency=1)


def test_budget_rules() -> None:
    budget = jev.RequestBudget(max_requests=1, max_tokens=100)
    assert budget.allows(100) and not budget.allows(101)
    budget.reserve(100)
    assert not budget.allows(0)
    budget.charge(100, 40)
    assert (budget.requests, budget.reserved_tokens, budget.tokens) == (1, 40, 40)
    with pytest.raises(ValueError):
        jev.RequestBudget(max_requests=-1, max_tokens=0)


def test_a_missing_api_key_is_refused() -> None:
    with pytest.raises(ValueError):
        jev.TypeSafeJevLabeller('', MODEL, max_text_chars=10)


def test_per_request_bound_follows_the_text_limit() -> None:
    config = LabelConfig.from_mapping({'FACTION_ANALYSIS_JEV_MAX_TEXT_CHARS': '100'})
    longest = pair(text='x' * 100, parent='x' * 100)
    assert jev.per_request_bound(config) == longest.token_bound()
    assert jev.new_budget(config).max_tokens == config.jev_max_input_tokens
