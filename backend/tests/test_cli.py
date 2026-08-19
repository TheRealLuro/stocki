"""The `stocki` command line: exit codes and readable output.

Exit codes matter because CI and the team both rely on them: 0 means the
database matches the files, anything else means look at the message.
"""

from dataclasses import replace

import pytest

from stocki.cli import main

from .conftest import make_rows, write_session

pytestmark = pytest.mark.db


def run(args, settings):
    return main(args, settings=settings)


@pytest.fixture
def data_dir(tmp_path):
    write_session(tmp_path, ticker="NVDA", day=1)
    write_session(tmp_path, ticker="NVDA", day=2)
    return tmp_path


def test_ingest_loads_the_files_and_exits_zero(db, db_settings, data_dir, capsys):
    code = run(["ingest", "--data-dir", str(data_dir)], db_settings)

    assert code == 0
    assert "156" in capsys.readouterr().out


def test_ingest_exits_non_zero_and_names_the_bad_file(db, db_settings, tmp_path, capsys):
    write_session(tmp_path, ticker="NVDA", day=1)
    write_session(tmp_path, ticker="TSLA", day=1, rows=make_rows(bars=12))

    code = run(["ingest", "--data-dir", str(tmp_path)], db_settings)

    assert code == 1
    assert "TSLA/day1.csv" in capsys.readouterr().out


def test_verify_passes_after_a_clean_ingest(db, db_settings, data_dir):
    run(["ingest", "--data-dir", str(data_dir)], db_settings)

    assert run(["verify", "--data-dir", str(data_dir)], db_settings) == 0


def test_verify_fails_when_nothing_was_loaded(db, db_settings, data_dir, capsys):
    code = run(["verify", "--data-dir", str(data_dir)], db_settings)

    assert code == 1
    assert "not loaded" in capsys.readouterr().out


def test_stats_prints_the_data_card(db, db_settings, data_dir, capsys):
    run(["ingest", "--data-dir", str(data_dir)], db_settings)

    code = run(["stats"], db_settings)

    output = capsys.readouterr().out
    assert code == 0
    assert "NVDA" in output
    assert "sessions" in output


def test_stats_on_an_empty_database_says_to_ingest(db, db_settings, capsys):
    code = run(["stats"], db_settings)

    assert code == 2
    assert "stocki ingest" in capsys.readouterr().err


def test_an_unreachable_database_prints_advice_not_a_traceback(db_settings, capsys):
    dead = replace(db_settings, host="127.0.0.1", port=1)

    code = run(["stats"], dead)

    captured = capsys.readouterr()
    assert code == 2
    assert "docker compose up" in captured.err
    assert "Traceback" not in captured.err
