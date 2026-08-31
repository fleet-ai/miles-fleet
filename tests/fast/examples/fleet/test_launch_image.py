import json

import pytest

from examples.fleet.launch import submit_run


def _payload(**overrides):
    payload = {
        "name": "miles-image-test",
        "command": "echo ready",
        "workers": 2,
        "gpus_per_worker": 1,
    }
    payload.update(overrides)
    return payload


def _load(tmp_path, payload):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(payload))
    return submit_run._load_payload(str(path))


def test_default_image_uses_ecr_digest_and_pull_secret(tmp_path):
    payload = _load(tmp_path, _payload())
    manifest = submit_run._render(payload)

    assert payload["image"] == submit_run.DEFAULT_IMAGE
    assert "dkr.ecr.us-east-1.amazonaws.com/fleet/miles-trainer@sha256:" in payload["image"]
    assert manifest.count(f"image: {submit_run.DEFAULT_IMAGE}") == 2
    assert manifest.count("name: ecr-pull") == 2
    assert "ghcr-pull" not in manifest


def test_custom_image_can_select_its_pull_secret(tmp_path):
    payload = _load(
        tmp_path,
        _payload(
            image="registry.example/research/trainer@sha256:" + "a" * 64,
            image_pull_secret="research-registry",
        ),
    )
    manifest = submit_run._render(payload)

    assert manifest.count("name: research-registry") == 2
    assert manifest.count(f"image: {payload['image']}") == 2


@pytest.mark.parametrize("field", ["image", "image_pull_secret"])
def test_registry_fields_reject_yaml_injection(tmp_path, field):
    with pytest.raises(SystemExit):
        _load(tmp_path, _payload(**{field: "valid\nimage: attacker.example/image:latest"}))
