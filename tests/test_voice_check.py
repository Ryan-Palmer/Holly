"""His voice said and heard back (app/voice_check.py): the comparing, without a voice."""

from app import voice_check as vc


def test_words_are_compared_as_said():
    assert vc.words("It's 20:39. Seventy-eight per cent, 3,000!") == ["it's", "20", "39", "78", "percent", "3000"]
    assert vc.wer("Rain. One hundred per cent chance of it.", "rain 100% chance of it") == 0.0
    assert vc.wer("Yeah?", "Yeah?") == 0.0
    assert vc.wer("Yeah?", "Ye.") == 1.0  # "Yeah?" alone, garbled


def test_a_scorecards_replies_each_once():
    card = dict(persona_rows=[dict(reply="Morning."), dict(reply="Morning.")],
                conversation_runs=[dict(run=1, turns=[dict(reply="Thrilling.")]), dict(run=2, turns=[dict(reply="No.")])],
                honesty_rows=[dict(run=1, reply="I don't know.")])
    assert vc.replies(card) == ["Morning.", "Thrilling.", "I don't know."]


def test_times_and_numbers_as_either_writes_them():
    """He writes "12:19" and "07:30"; Whisper hears "12.19" and "7.30"."""
    assert vc.wer("12:19. Not that I'm counting.", "12.19 not that I'm counting") == 0.0
    assert vc.wer("Done. Due at 07:30 tomorrow.", "Done. Due at 7.30 tomorrow.") == 0.0


def test_prices_as_either_writes_them():
    assert vc.words("One pound eighty.") == vc.words("£1.80") == ["1", "pound", "80"]
    assert vc.wer("A large letter's three pounds thirty.", "A large letter's £3.30.") == 0.0
