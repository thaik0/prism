from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TERRAFORM = ROOT / "infra" / "terraform"
WORKFLOWS = ROOT / ".github" / "workflows"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_remote_state_uses_native_s3_locking_and_ignores_local_state() -> None:
    backend = _read(TERRAFORM / "main" / "backend.tf")
    ignored = _read(ROOT / ".gitignore")
    assert 'backend "s3"' in backend
    assert "use_lockfile = true" in backend
    assert "encrypt      = true" in backend
    assert "dynamodb" not in backend.lower()
    assert "*.tfstate" in ignored
    assert "**/.terraform/" in ignored
    assert "*.tfplan" in ignored


def test_batch_definition_preserves_phase2_and_requires_digest_promotion() -> None:
    batch = _read(TERRAFORM / "main" / "batch.tf")
    variables = _read(TERRAFORM / "main" / "variables.tf")
    assert 'cpuArchitecture       = "ARM64"' in batch
    assert '{ type = "VCPU", value = "1" }' in batch
    assert '{ type = "MEMORY", value = "2048" }' in batch
    assert 'attempts = 1' in batch
    assert 'logDriver = "awslogs"' in batch
    assert 'command          = ["prism-cloud-bootstrap"]' in batch
    assert 'count = var.image_uri == null ? 0 : 1' in batch
    assert "@sha256:" in variables


def test_oidc_trust_is_repository_audience_and_environment_exact() -> None:
    oidc = _read(TERRAFORM / "main" / "oidc.tf")
    variables = _read(TERRAFORM / "main" / "variables.tf")
    assert 'values   = ["sts.amazonaws.com"]' in oidc
    assert 'repo:${var.github_repository}:environment:' in oidc
    assert 'default     = "thaik0/prism"' in variables
    assert 'default     = "prism-cloud-deploy"' in variables
    assert 'default     = "prism-cloud-apply"' in variables
    assert "AdministratorAccess" not in oidc


def test_workflows_use_oidc_and_keep_publish_promote_smoke_separate() -> None:
    publish = _read(WORKFLOWS / "cloud-image-publish.yml")
    terraform = _read(WORKFLOWS / "cloud-terraform.yml")
    smoke = _read(WORKFLOWS / "cloud-smoke.yml")
    ci = _read(WORKFLOWS / "cloud-ci.yml")
    combined = "\n".join((publish, terraform, smoke, ci)).lower()

    assert "id-token: write" in publish
    assert "id-token: write" in terraform
    assert "id-token: write" in smoke
    assert "aws-access-key-id" not in combined
    assert "aws-secret-access-key" not in combined
    assert "register-job-definition" not in publish
    assert "batch:register" not in publish.lower()
    assert "terraform -chdir=infra/terraform/main apply" in terraform
    assert "prism-cloud submit" in smoke
    assert "prism-cloud wait" in smoke
    assert "prism-cloud download" in smoke
    assert "prism-container-verify" in smoke
    assert ":latest" not in combined
