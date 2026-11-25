"""Listening: which lines are for Holly (his name near an end, or said while he's awake),
and how long the mic stays deaf after he speaks."""

import pytest

from app.listen import ECHO_TAIL_S, after_speaking, for_holly

WAKE = {"holly", "hollie", "holley"}  # holly.toml's wake_words: how Whisper has spelt him
EDGE = 3  # holly.toml's wake_edge_words
ASLEEP = (100.0, 50.0)  # (started, awake_until): begun after his awake time ran out


@pytest.mark.parametrize("text", [
    "Holly, what's the weather in Keswick?",
    "Hey Holly, what's for tea?",
    "Oi, hollie, what time is it?",
    "What's the weather tomorrow, Holly?",
    "Is the lasagne done yet, Holly?",
    "Put the kettle on, would you, Holley?",
    "Holly's got a timer on for that.",  # "Holly's" is his name too
    "Right then, Holly, what's for tea tonight?",  # the third word
    "Is the lasagne done, Holly, or not?",  # the third from the end
])
def test_his_name_in_the_first_or_last_words_wakes_him(text):
    assert for_holly(text, *ASLEEP, WAKE, EDGE)


@pytest.mark.parametrize("text", [
    "Jo said that Holly was being sarcastic about the lasagne.",  # mid-sentence: talk about him
    "I bought some holly for the door at the market.",
    "Is the lasagne done yet, Holly, or not yet?",  # the fourth from the end
    "Sam and June are coming round for lasagne tonight.",  # no name at all
])
def test_his_name_mid_sentence_or_not_at_all_doesnt(text):
    assert not for_holly(text, *ASLEEP, WAKE, EDGE)


@pytest.mark.parametrize("started, awake_until, answered", [
    (120.0, 130.0, True),  # awake: carrying on, no name needed
    (130.0, 130.0, True),  # begun just as it ran out
    (130.5, 130.0, False),  # asleep: it is
])
def test_while_hes_awake_no_name_is_needed(started, awake_until, answered):
    assert for_holly("And what about tomorrow?", started, awake_until, WAKE, EDGE) is answered
    assert for_holly("And what about tomorrow, Holly?", started, awake_until, WAKE, EDGE)


def test_with_no_wake_words_he_answers_everything():
    assert for_holly("Sam and June are coming round tonight.", *ASLEEP, set(), EDGE)


@pytest.mark.parametrize("now, last_played, spoke_until", [
    (10.0, None, 10.0),  # no word from the speakers on when it played: now
    (10.0, 9.2, 10.0),  # played long since: now
    # His last sample yet to leave the speakers (the face's audio is ~0.6 s behind): deaf till
    # it has, plus the echo tail. Deaf only to now plus the tail, the mic caught his last
    # words, and he answered them.
    (10.0, 10.6, 10.6),
])
def test_deaf_until_his_last_sample_has_played_plus_the_echo_tail(now, last_played, spoke_until):
    deaf_until, awake_until = after_speaking(now, last_played, 30.0)
    assert deaf_until == pytest.approx(spoke_until + ECHO_TAIL_S)
    assert awake_until == pytest.approx(spoke_until + 30.0)  # awake for awake_s from the same moment
