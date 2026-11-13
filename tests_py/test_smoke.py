import balatro_rl
from balatro_rl import _engine


def test_engine_version_matches_package():
    assert _engine.version() == "0.1.0"
    assert balatro_rl.version() == _engine.version()
