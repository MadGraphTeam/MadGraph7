"""Select the finite current CLI authority and the three separate failed pilots."""
import hashlib
import json
from pathlib import Path

ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
RUN = HERE.parent
BASE = RUN.parents[1]
PRIVATE = ROOT / ".benchmark-work/profile-cli-final-publication-v1"
PRIVATE.mkdir(exist_ok=False)
selected = {}


def select(path, category):
    path = Path(path)
    assert path.is_file() and not path.is_symlink(), path
    relative = str(path.relative_to(ROOT))
    if relative in selected:
        return
    data = path.read_bytes()
    assert not data.startswith((b"\x7fELF", b"!<arch>\n")), path
    assert path.suffix not in (".o", ".a", ".so", ".mod", ".pyc"), path
    selected[relative] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "category": category}


omitted = {HERE / "validated-records.json", HERE / "result.json"}
for path in RUN.rglob("*"):
    if path.is_file() and path not in omitted:
        select(path, "actual-v4-72-session-authority")
for path in (BASE / "profile-cli-expanded-plan-v4").rglob("*"):
    if path.is_file():
        select(path, "executed-v4-plan")
for path in (BASE / "profile-cli-expanded-review-v4").rglob("*"):
    if path.is_file():
        select(path, "independent-v4-plan-review")

for name in ("profile-cli-expanded", "profile-cli-expanded-v2", "profile-cli-expanded-v3"):
    directory = BASE / name
    for path in directory.rglob("*"):
        if path.is_file():
            select(path, "historical-failed-pilot-" + name)
    result = json.loads((directory / "run-v1/qcd-1/6/0/result.json").read_text())
    path = ROOT / result["specification"]["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == result["specification"]["sha256"]
    select(path, "historical-attempted-specification-only")
for name in ("profile-cli-expanded-review-v1", "profile-cli-expanded-review-v2", "profile-cli-expanded-review-v3"):
    for path in (BASE / name).rglob("*"):
        if path.is_file():
            select(path, "historical-independent-plan-review")

inventory_path = RUN / "publication-inventory.json"
assert not inventory_path.exists()
inventory = {"schema": "finite-cli-profile-publication-inventory-v1", "scope": "Executed v4 plan, all72 strict descriptors/raw/controller histories, deduplicated generated-source closure and separately retained failed predecessors", "optional_local_omissions": [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "reason": "Optional mirrored reader-return array/draft summary; complete raw authority and concise summary retained"} for p in sorted(omitted)], "excluded_historical_drafts": "Unused 48-card sets, original old plan manifests/readiness drafts and redundant old software proof capsules are not runtime inputs to retained failure_snapshot; original attempted specification and complete result.files remain selected", "native_payloads_selected": False, "required_scientific_cache_exception": "Named SavedStabilityRun_Ninja/NinjaRS_0.pkl are exact primitive-only scientific inputs checked by existing reader; they are retained, including raw inventory duplicates", "files": dict(sorted(selected.items())), "files_count": len(selected), "payload_bytes": sum(v["bytes"] for v in selected.values())}
inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
stage = PRIVATE / "stage-files.txt"
stage.write_text("\n".join(sorted([*selected, str(inventory_path.relative_to(ROOT))])) + "\n")
summary = {"inventory": str(inventory_path.relative_to(ROOT)), "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(), "payload_files": len(selected), "payload_bytes": inventory["payload_bytes"], "stage_files": str(stage.relative_to(ROOT)), "stage_files_sha256": hashlib.sha256(stage.read_bytes()).hexdigest(), "stage_paths": len(selected)+1}
(PRIVATE / "selection.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary))
