from __future__ import annotations

import json

from mean_field.devtools._runtime import parse_csv_floats, parse_csv_ints, write_json


def test_devtools_write_json_reuses_shared_artifact_writer(tmp_path) -> None:
    path = tmp_path / "nested" / "payload.json"

    write_json(path, {"b": 2, "a": 1})

    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert not path.with_name(path.name + ".tmp").exists()


def test_devtools_csv_parsers_keep_existing_behavior() -> None:
    assert parse_csv_floats("1.0, 2.5") == (1.0, 2.5)
    assert parse_csv_ints("1, 2,3") == (1, 2, 3)
