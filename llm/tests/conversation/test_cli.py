from __future__ import annotations

import json

from llm.src.conversation.cli import main


def test_cli_smoke_is_fully_stubbed(tmp_path, monkeypatch, capsys):
    def fake_call_lm_studio(api_base, payload, timeout_s):
        del timeout_s
        assert api_base == "http://localhost:1234"
        assert payload["model"] == "qwen/qwen3-14b"
        assert payload["temperature"] == 0.0
        assert payload["top_p"] == 1.0
        assert payload["max_tokens"] == 512
        assert payload["stream"] is False
        assert payload["response_format"]["type"] == "json_schema"
        return {
            "ok": True,
            "status_code": 200,
            "error": None,
            "response_json": {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": (
                                    '{"task":"extract_cf_request","status":"complete","cf_request":'
                                    '{"Income":140,"Family":2,"CCAvg":7.7376709303,"Education":2,"Mortgage":32,'
                                    '"SecuritiesAccount":1,"CDAccount":1,"Online":1,"CreditCard":0},'
                                    '"missing_fields":[],"conflicts":[],"notes":[]}'
                                ),
                            }
                        ],
                    }
                ]
            },
            "response_text": '{"ok":true}',
            "elapsed_ms": 10.0,
        }

    monkeypatch.setattr("llm.src.conversation.parser_adapter.call_lm_studio", fake_call_lm_studio)

    rc = main(
        [
            "--text",
            "Income 140, Family 2, CCAvg 7.7376709303, Education 2, Mortgage 32, SecuritiesAccount 1, CDAccount 1, Online 1, CreditCard 0.",
            "--out-dir",
            str(tmp_path),
            "--scenario-slug",
            "cli_smoke",
        ]
    )

    assert rc == 0
    stdout = capsys.readouterr().out
    assert "desired outcome" in stdout

    output_dir = next(tmp_path.iterdir())
    manifest = json.loads((output_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "RUNTIME_SUCCESS"
