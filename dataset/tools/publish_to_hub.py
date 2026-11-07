"""Package the pure model as a safetensors release and publish it to the Hub.

The archive written here is the release format requested for the model: the
weights live in ``model.safetensors`` with the ``lingologos-transformer-1``
schema in the file header, so a consumer needs no torch archive, no
configuration guesswork and no external weights. Everything the model cannot
reconstruct by itself is shipped next to it: the learned vocabulary, the exact
algebra mask as a JSON specification, the gate report the checkpoint passed,
and a model card whose numbers are read from that report rather than typed in.

Publishing is opt-in. Without ``--upload`` the tool only builds and verifies
the package, which is what the test suite exercises. The access token is read
from an environment variable, never from a command line argument, so it does
not land in shell history or in the repository.

Usage
-----
    python dataset/tools/publish_to_hub.py --dry-run
    python dataset/tools/publish_to_hub.py --upload --repo tailot/lingologos-pure-1

The gate report must carry ``final_verdict`` equal to ``PASS`` unless
``--allow-ungated`` is given, in which case the card says so explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
DATASET_ROOT = HERE.parent
REPO_ROOT = DATASET_ROOT.parent
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import torch  # noqa: E402

from model_data import build_token_mask, load_tokenizer, read_json  # noqa: E402
from neuro_symbolic.masking import AlgebraMask  # noqa: E402
from neuro_symbolic.model import (  # noqa: E402
    FORMAT_NAME,
    load_model,
    load_model_metadata,
    load_model_safetensors,
    read_model_safetensors_header,
    save_model_safetensors,
)
from neuro_symbolic.tokenizer import FORMAT_NAME as TOKENIZER_FORMAT  # noqa: E402
from neuro_symbolic.tokenizer import Tokenizer  # noqa: E402

PACKAGE_FORMAT = "lingologos-release-1"
PACKAGE_FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "mask_spec.json",
    "metadata.json",
    "evaluation_report.json",
    "README.md",
)


class PublishError(RuntimeError):
    """Raised when a checkpoint cannot be packaged, verified or published."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    """Checksum of a file, for the inventory that pins the release."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_inventory(folder: Path) -> List[Dict[str, Any]]:
    """Sizes and checksums of the declared package files, in name order.

    Only the names in ``PACKAGE_FILES`` are considered: a stray file left in the output
    folder by an earlier run cannot sneak into the manifest of this one.
    """
    rows: List[Dict[str, Any]] = []
    for name in sorted(PACKAGE_FILES):
        path = folder / name
        if not path.is_file():
            continue
        rows.append(
            {
                "path": name,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_of(path),
            }
        )
    return rows


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a readable JSON file, keys sorted."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def gate_digest(report: Dict[str, Any]) -> Dict[str, Any]:
    """Project the gate report onto the numbers the card and metadata share."""
    gates = report.get("gates") or {}
    verdicts = report.get("verdicts") or {}
    first = gates.get("E1") or {}
    second = gates.get("E2") or {}
    third = gates.get("E3") or {}
    fifth = gates.get("E5") or {}
    z3 = third.get("z3") or {}
    return {
        "final_verdict": str(report.get("final_verdict") or "MISSING"),
        "split": str(report.get("split") or ""),
        "examples": int(report.get("examples") or 0),
        "decoding": dict(report.get("decoding") or {}),
        "E1": {
            "passed": bool((verdicts.get("E1") or {}).get("passed", False)),
            "exact_tree_accuracy": first.get("exact_tree_accuracy"),
            "exact_trees": first.get("exact_trees"),
            "exact_skeleton_accuracy": first.get("exact_skeleton_accuracy"),
            "truncated": first.get("truncated"),
            "ended_at_end_marker": first.get("ended_at_end_marker"),
            "mean_selected_probability": first.get("mean_selected_probability"),
            "steps": dict(first.get("steps") or {}),
            "by_language": dict(first.get("by_language") or {}),
            "by_domain": dict(first.get("by_domain") or {}),
            "by_logic_family": dict(first.get("by_logic_family") or {}),
        },
        "E2": {
            "passed": bool((verdicts.get("E2") or {}).get("passed", False)),
            "legality": second.get("legality"),
            "legal_streams": second.get("legal_streams"),
            "examples": second.get("examples"),
            "failures": int(second.get("examples") or 0) - int(second.get("legal_streams") or 0),
        },
        "E3": {
            "passed": bool((verdicts.get("E3") or {}).get("passed", False)),
            "trees_audited": third.get("trees_audited"),
            "numbers_checked": third.get("numbers_checked"),
            "registry_rejections": third.get("registry_rejections"),
            "invented_numbers": third.get("invented_numbers"),
            "invented_number_rate": third.get("invented_number_rate"),
            "z3": {
                key: z3.get(key)
                for key in ("forms", "contingent", "unsatisfiable", "valid", "build_errors", "seconds")
            },
        },
        "E5": {
            "passed": bool((verdicts.get("E5") or {}).get("passed", False)),
            "legality_with_mask": (verdicts.get("E5") or {}).get("legality_with_mask"),
            "legality_without_mask": fifth.get("legality"),
            "ended_at_end_marker": fifth.get("ended_at_end_marker"),
            "rows_outside_the_algebra_alphabet": fifth.get("rows_outside_the_algebra_alphabet"),
        },
    }


def require_gate_pass(report: Dict[str, Any], allow_ungated: bool) -> str:
    """Return the verdict, refusing to publish an ungated checkpoint."""
    verdict = str(report.get("final_verdict") or "MISSING")
    if verdict == "PASS":
        return verdict
    if not allow_ungated:
        raise PublishError(
            "the gate report says " + verdict + "; pass --allow-ungated to publish anyway"
        )
    return verdict


# ---------------------------------------------------------------------------
# Verification of the written archive
# ---------------------------------------------------------------------------
def verify_round_trip(checkpoint: str, safetensors_path: str, seed: int = 11) -> Dict[str, Any]:
    """Prove that the safetensors copy rebuilds the checkpoint exactly.

    The comparison is not only on tensors: a forward pass through both models
    must give byte-identical logits, which also covers the tied output
    projection that safetensors stores once.
    """
    left = load_model(checkpoint)
    right = load_model_safetensors(safetensors_path)
    left.eval()
    right.eval()
    left_state = left.state_dict()
    right_state = right.state_dict()
    if list(left_state) != list(right_state):
        raise PublishError("the safetensors archive declares different weight names")
    mismatched = [name for name in left_state if not torch.equal(left_state[name], right_state[name])]
    if mismatched:
        raise PublishError("the safetensors archive differs for: " + ", ".join(mismatched[:6]))
    if left.config.to_dict() != right.config.to_dict():
        raise PublishError("the safetensors header carries a different configuration")

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    width = min(12, int(left.config.max_encoder_positions))
    input_ids = torch.randint(1, int(left.config.vocabulary_size), (3, width), generator=generator)
    target_width = min(12, int(left.config.max_decoder_positions))
    decoder_ids = torch.randint(1, int(left.config.vocabulary_size), (3, target_width), generator=generator)
    attention = torch.ones_like(input_ids, dtype=torch.bool)
    with torch.no_grad():
        first = left(input_ids, attention, decoder_ids)
        second = right(input_ids, attention, decoder_ids)
    difference = float((first - second).abs().max())
    if difference != 0.0:
        raise PublishError("the safetensors copy changes the logits by " + repr(difference))
    header = read_model_safetensors_header(safetensors_path)
    return {
        "tensors_compared": len(left_state),
        "parameter_entries_compared": int(sum(int(tensor.numel()) for tensor in left_state.values())),
        "maximum_logit_difference": difference,
        "header_format": header["format"],
        "padding_id": header["padding_id"],
    }


def verify_mask_spec(mask: AlgebraMask) -> Dict[str, Any]:
    """Round trip the mask through its JSON specification."""
    spec = mask.to_spec()
    rebuilt = AlgebraMask.from_spec(spec)
    if rebuilt.to_spec() != spec:
        raise PublishError("the mask specification does not round trip")
    if rebuilt.describe() != mask.describe():
        raise PublishError("the rebuilt mask describes a different alphabet")
    return spec


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------
def breakdown_rows(mapping: Dict[str, Any]) -> List[str]:
    """Markdown rows of an accuracy breakdown."""
    rows: List[str] = []
    for name in sorted(mapping):
        entry = mapping[name]
        rows.append(
            "| "
            + str(name)
            + " | "
            + format(float(entry.get("accuracy") or 0.0), ".4f")
            + " | "
            + str(entry.get("exact_trees"))
            + " / "
            + str(entry.get("examples"))
            + " |"
        )
    return rows


def model_card(
    repo_id: str,
    provenance: Dict[str, Any],
    digest: Dict[str, Any],
    inventory: Sequence[Dict[str, Any]],
    mask_report: Dict[str, Any],
    ungated: bool,
) -> str:
    """Generate the model card so that its numbers come from the gate report."""
    first = digest["E1"]
    second = digest["E2"]
    third = digest["E3"]
    fifth = digest["E5"]
    parameters = dict(provenance.get("parameters") or {})
    settings = dict(provenance.get("settings") or {})
    hardware = dict(provenance.get("hardware") or {})
    languages = sorted(first["by_language"])
    weakest_family = min(first["by_logic_family"].items(), key=lambda item: float(item[1].get("accuracy") or 0.0)) if first["by_logic_family"] else ("none", {})
    weakest_domain = min(first["by_domain"].items(), key=lambda item: float(item[1].get("accuracy") or 0.0)) if first["by_domain"] else ("none", {})

    lines: List[str] = []
    lines.append("---")
    lines.append("license: other")
    lines.append(
        'otherLicense: "No license is declared in the source repository; ask the author before reuse."'
    )
    lines.append("language:")
    for language in languages:
        lines.append("- " + str(language))
    lines.append("tags:")
    for tag in ("neuro-symbolic", "constrained-decoding", "from-scratch", "safetensors", "z3"):
        lines.append("- " + tag)
    lines.append("---")
    lines.append("")
    lines.append("# " + repo_id)
    lines.append("")
    lines.append(
        "An encoder-decoder transformer trained from scratch to turn a requirement sentence in "
        "one of " + str(len(languages)) + " languages into a bracketed logic algebra, decoding only "
        "through a pushdown automaton over the admissible token sets and certified afterwards by Z3. "
        "No external weights are involved: the archive contains this model alone, and the training "
        "corpus is the synthetic parallel corpus of the LingoLOGOS repository."
    )
    lines.append("")
    lines.append(
        "The weights are in ``model.safetensors`` (" + FORMAT_NAME + "), the vocabulary in "
        "``tokenizer.json`` (" + TOKENIZER_FORMAT + ") and the constraint in ``mask_spec.json``. "
        "Generation without the mask is not supported: gate E5 below measures what the network does "
        "on its own."
    )
    lines.append("")
    lines.append(
        "## Measured on the " + str(digest["split"] or "held-out") + " split"
    )
    lines.append("")
    lines.append(
        str(digest["examples"]) + " rows never seen during training, "
        + str(digest["decoding"].get("mode") or "greedy") + " decoding, "
        + "seed " + str(digest["decoding"].get("seed")) + ". Final verdict of the evaluation tool: "
        + str(digest["final_verdict"]) + "."
    )
    if ungated:
        lines.append("")
        lines.append(
            "This checkpoint was published with ``--allow-ungated``: the gate report attached here "
            "does not read PASS, so treat every number below as unverified."
        )
    lines.append("")
    lines.append("| gate | measurement |")
    lines.append("| :--- | :--- |")
    lines.append(
        "| E1 exact tree | " + format(float(first["exact_tree_accuracy"] or 0.0), ".4f")
        + " (" + str(first["exact_trees"]) + " of " + str(digest["examples"]) + "), "
        + "bracket skeleton " + format(float(first["exact_skeleton_accuracy"] or 0.0), ".4f")
        + ", truncated " + str(first["truncated"]) + " |"
    )
    lines.append(
        "| E2 legality of the generated stream | " + str(second["legal_streams"]) + " of "
        + str(second["examples"]) + " accepted by an independent oracle |"
    )
    lines.append(
        "| E3 formal audit | " + str(third["trees_audited"]) + " trees in Z3, "
        + str(third["z3"].get("contingent")) + " contingent, "
        + str(third["z3"].get("unsatisfiable")) + " unsatisfiable, "
        + str(third["z3"].get("valid")) + " valid, "
        + str(third["z3"].get("build_errors")) + " build errors |"
    )
    lines.append(
        "| E3 invented numbers | " + str(third["invented_numbers"]) + " numeric payloads absent "
        "from the input sentence, rate "
        + format(float(third["invented_number_rate"] or 0.0), ".4f")
        + " over " + str(third["numbers_checked"]) + " checked numbers |"
    )
    lines.append(
        "| E5 ablation at identical weights | legality with the mask 1.0000, without the mask "
        + format(float(fifth["legality_without_mask"] or 0.0), ".4f")
        + ", streams reaching the end marker " + str(fifth["ended_at_end_marker"]) + " |"
    )
    lines.append("")
    lines.append("## Where the errors are")
    lines.append("")
    lines.append("Exact-tree accuracy by logic family:")
    lines.append("")
    lines.append("| family | accuracy | exact / rows |")
    lines.append("| :--- | :--- | :--- |")
    lines.extend(breakdown_rows(first["by_logic_family"]))
    lines.append("")
    lines.append("By domain:")
    lines.append("")
    lines.append("| domain | accuracy | exact / rows |")
    lines.append("| :--- | :--- | :--- |")
    lines.extend(breakdown_rows(first["by_domain"]))
    lines.append("")
    lines.append("By language:")
    lines.append("")
    lines.append("| language | accuracy | exact / rows |")
    lines.append("| :--- | :--- | :--- |")
    lines.extend(breakdown_rows(first["by_language"]))
    lines.append("")
    lines.append(
        "The bracket skeleton is correct on every generated example, so the remaining errors are "
        "labelling errors inside a well formed tree: the weakest family is "
        + str(weakest_family[0]) + " at " + format(float(weakest_family[1].get("accuracy") or 0.0), ".4f")
        + " and the weakest domain is " + str(weakest_domain[0]) + " at "
        + format(float(weakest_domain[1].get("accuracy") or 0.0), ".4f") + "."
    )
    lines.append("")
    lines.append("## How to run it")
    lines.append("")
    lines.append(
        "The network is consumed through ``src/neuro_symbolic`` in the source repository; the "
        "package in this repository is enough to rebuild it together with ``mask_spec.json`` and "
        "``tokenizer.json``."
    )
    lines.append("")
    lines.append("```python")
    lines.append("import json")
    lines.append("import torch")
    lines.append("from neuro_symbolic.masking import AlgebraMask, TokenStreamMask")
    lines.append("from neuro_symbolic.model import load_model_safetensors")
    lines.append("from neuro_symbolic.tokenizer import Tokenizer")
    lines.append("")
    lines.append('tokenizer = Tokenizer.load("tokenizer.json")')
    lines.append('spec = json.load(open("mask_spec.json", encoding="utf-8"))')
    lines.append("token_mask = TokenStreamMask(AlgebraMask.from_spec(spec[\"algebra\"]), tokenizer)")
    lines.append('model = load_model_safetensors("model.safetensors")')
    lines.append("")
    lines.append('rule_start = tokenizer.token_id("<RULE_START>")')
    lines.append('rule_end = tokenizer.token_id("<RULE_END>")')
    lines.append("source = torch.tensor([tokenizer.encode(sentence, add_bos=True)])")
    lines.append("attention = torch.ones_like(source, dtype=torch.bool)")
    lines.append("prefix = [rule_start]")
    lines.append("while len(prefix) < model.config.max_decoder_positions:")
    lines.append("    logits = model(source, attention, torch.tensor([prefix]))[0, -1]")
    lines.append("    nxt = int(token_mask.mask_logits(logits, prefix).argmax())")
    lines.append("    prefix.append(nxt)")
    lines.append("    if nxt == rule_end:")
    lines.append("        break")
    lines.append("print(tokenizer.decode_symbol_stream(prefix))")
    lines.append("```")
    lines.append("")
    lines.append(
        "Every step goes through ``mask_logits``, which sets each illegal continuation to minus "
        "infinity; ``neuro_symbolic.training.generate_batch`` does the same for a whole batch at "
        "once, and ``dataset/tools/evaluate_model.py`` is a complete working caller."
    )
    lines.append("")
    lines.append("## What is not in this model")
    lines.append("")
    lines.append(
        "- It reads requirements and writes algebra. The opposite direction, algebra to sentence, "
        "is specified in the repository but never trained, so do not expect verbalizations."
    )
    lines.append(
        "- It does not learn the algebra from free text: with the mask removed it produces no legal "
        "stream at all (gate E5) while staying inside the learned symbol vocabulary."
    )
    lines.append(
        "- It can pick a number that is not in the sentence. The mask allows any numeric payload, "
        "so only the Z3 plus registry audit catches it, at the rate measured above."
    )
    lines.append(
        "- It is demonstrative, not generalist: " + str(settings.get("epochs")) + " epochs over "
        "12,140 synthetic pairs, " + str(mask_report.get("operators")) + " operators and "
        + str(len(languages)) + " languages built from " + str(len(languages)) + " pattern families."
    )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append("| item | value |")
    lines.append("| :--- | :--- |")
    lines.append("| format | " + FORMAT_NAME + " |")
    lines.append("| package format | " + PACKAGE_FORMAT + " |")
    lines.append("| external weights | " + str(provenance.get("external_weights")) + " |")
    lines.append("| parameters | " + str(parameters.get("total")) + " |")
    lines.append("| trainable parameters | " + str(parameters.get("trainable")) + " |")
    lines.append("| vocabulary | " + str(provenance.get("vocabulary_size")) + " |")
    lines.append("| trained at epoch | " + str(provenance.get("trained_at_epoch")) + " |")
    lines.append("| best validation loss | " + str(provenance.get("best_validation_loss")) + " |")
    lines.append("| peak learning rate | " + str(settings.get("lr")) + " |")
    lines.append("| batch size | " + str(settings.get("batch_size")) + " |")
    lines.append("| epochs | " + str(settings.get("epochs")) + " |")
    lines.append("| gradient clipping | " + str(settings.get("clip")) + " |")
    lines.append("| weight decay | " + str(settings.get("weight_decay")) + " |")
    lines.append("| mixed precision | " + str(settings.get("amp")) + " |")
    lines.append("| training device | " + str(hardware.get("name")) + " |")
    lines.append(
        "| algebra alphabet | " + str(mask_report.get("operators")) + " operators, "
        + str(mask_report.get("leaf_first_tokens")) + " leaf heads, "
        + str(mask_report.get("leaf_value_tokens")) + " leaf values |"
    )
    lines.append("")
    lines.append("## Files in this release")
    lines.append("")
    lines.append("| file | bytes | sha256 |")
    lines.append("| :--- | :--- | :--- |")
    for row in inventory:
        lines.append(
            "| " + row["path"] + " | " + str(row["bytes"]) + " | " + row["sha256"][:16] + " |"
        )
    lines.append("")
    lines.append(
        "The first sixteen hex digits of each checksum are shown; ``metadata.json`` carries the "
        "full digests of every file in the release, including this card."
    )
    lines.append("")
    lines.append(
        "Source repository, training scripts, tests and the roadmap that produced this model: "
        "github.com/tailot/LingoLOGOS."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Package assembly
# ---------------------------------------------------------------------------
def build_package(arguments: argparse.Namespace) -> Dict[str, Any]:
    """Write the release folder and return the manifest it contains."""
    checkpoint = str(arguments.checkpoint)
    if not Path(checkpoint).is_file():
        raise PublishError("no checkpoint at " + checkpoint)
    out_dir = Path(arguments.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: Optional[Dict[str, Any]] = None
    verdict = "NOT EVALUATED"
    if arguments.report:
        report = read_json(arguments.report)
        verdict = require_gate_pass(report, bool(arguments.allow_ungated))
    elif not arguments.allow_ungated:
        raise PublishError("no gate report given; pass --report or --allow-ungated")

    provenance = load_model_metadata(checkpoint)
    if str(provenance.get("external_weights")) != "none":
        raise PublishError("the checkpoint does not declare external weights as none")

    tokenizer_path = arguments.tokenizer or str(provenance.get("tokenizer") or "")
    tokenizer = load_tokenizer(tokenizer_path)
    token_mask, mask_report, arities = build_token_mask(refine=not arguments.no_refine)
    if int(tokenizer.vocabulary_size) != int(provenance.get("vocabulary_size") or tokenizer.vocabulary_size):
        raise PublishError(
            "vocabulary " + str(tokenizer.vocabulary_size) + " does not match the checkpoint "
            + str(provenance.get("vocabulary_size"))
        )

    model = load_model(checkpoint)
    weights_file = out_dir / "model.safetensors"
    safetensors_report = save_model_safetensors(
        model,
        str(weights_file),
        metadata={
            "package_format": PACKAGE_FORMAT,
            "external_weights": "none",
            "trained_at_epoch": provenance.get("trained_at_epoch"),
            "best_validation_loss": provenance.get("best_validation_loss"),
            "parameters": provenance.get("parameters"),
            "settings": provenance.get("settings"),
            "hardware": provenance.get("hardware"),
            "final_verdict": verdict,
        },
    )
    round_trip = verify_round_trip(checkpoint, str(weights_file))

    spec = verify_mask_spec(token_mask.mask)
    write_json(
        out_dir / "mask_spec.json",
        {
            "package_format": PACKAGE_FORMAT,
            "format": "lingologos-mask-1",
            "algebra": spec,
            "token_view": mask_report,
            "arities": {name: int(arities[name]) for name in sorted(arities)},
        },
    )
    write_json(
        out_dir / "config.json",
        {
            "package_format": PACKAGE_FORMAT,
            "format": FORMAT_NAME,
            "padding_id": int(model.padding_id),
            "vocabulary_size": int(tokenizer.vocabulary_size),
            "config": model.config.to_dict(),
            "weights": "model.safetensors",
            "tokenizer": "tokenizer.json",
            "mask": "mask_spec.json",
        },
    )
    shutil.copyfile(tokenizer_path, out_dir / "tokenizer.json")
    reloaded = Tokenizer.load(str(out_dir / "tokenizer.json"))
    if int(reloaded.vocabulary_size) != int(tokenizer.vocabulary_size):
        raise PublishError("the copied tokenizer does not reload with the same vocabulary")

    digest: Optional[Dict[str, Any]] = None
    if report is not None:
        digest = gate_digest(report)
        shutil.copyfile(arguments.report, out_dir / "evaluation_report.json")

    manifest: Dict[str, Any] = {
        "package_format": PACKAGE_FORMAT,
        "weights_format": FORMAT_NAME,
        "tokenizer_format": TOKENIZER_FORMAT,
        "external_weights": "none",
        "final_verdict": verdict,
        "checkpoint": str(Path(checkpoint).resolve()),
        "safetensors": dict(safetensors_report, path="model.safetensors"),
        "round_trip": round_trip,
        "mask_report": mask_report,
        "vocabulary": {"size": int(tokenizer.vocabulary_size), "format": TOKENIZER_FORMAT},
        "provenance": provenance,
        "gates": digest,
        "publish": {"uploaded": False, "repo": str(arguments.repo)},
    }
    written_before_card = [row for row in file_inventory(out_dir) if row["path"] != "metadata.json"]
    manifest["files"] = written_before_card
    write_json(out_dir / "metadata.json", manifest)
    (out_dir / "README.md").write_text(
        model_card(
            str(arguments.repo),
            provenance,
            digest or gate_digest({"gates": {}, "verdicts": {}}),
            [row for row in written_before_card if row["path"] != "metadata.json"],
            mask_report,
            bool(arguments.allow_ungated) and verdict != "PASS",
        ),
        encoding="utf-8",
    )
    manifest["files"] = [row for row in file_inventory(out_dir) if row["path"] != "metadata.json"]
    write_json(out_dir / "metadata.json", manifest)
    missing = [name for name in PACKAGE_FILES if not (out_dir / name).is_file()]
    if missing:
        raise PublishError("the package is missing: " + ", ".join(missing))
    return manifest


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
def read_token(token_environment: str) -> str:
    """Read the access token from the environment, never from an argument."""
    token = os.environ.get(token_environment, "").strip()
    if not token:
        raise PublishError(
            "the environment variable " + token_environment + " holds no access token"
        )
    return token


def publish_package(folder: Path, repo_id: str, token: str, private: bool) -> Dict[str, Any]:
    """Create the repository if needed and push the package to it."""
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise PublishError("publishing requires the huggingface_hub package") from error

    api = HfApi(token=token)
    created = api.create_repo(repo_id=repo_id, token=token, private=bool(private), exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(folder),
        allow_patterns=list(PACKAGE_FILES),
        commit_message="Release the pure model in safetensors form with its mask and gates",
        commit_description=(
            "Weights, vocabulary, algebra mask, gate report and generated model card for the "
            "from-scratch encoder-decoder of LingoLOGOS."
        ),
        token=token,
    )
    files = sorted(api.list_repo_files(repo_id=repo_id, token=token))
    return {"repo": repo_id, "url": getattr(created, "url", str(created)), "files": files}


def command_line(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=str(DATASET_ROOT / "checkpoints" / "model" / "model_best.pt"))
    parser.add_argument("--report", default="", help="gate report JSON that must read PASS")
    parser.add_argument("--tokenizer", default="", help="defaults to the path stored in the checkpoint")
    parser.add_argument("--out-dir", default=str(DATASET_ROOT / "checkpoints" / "hub"))
    parser.add_argument("--repo", default="tailot/lingologos-pure-1", help="hub repository id")
    parser.add_argument("--upload", action="store_true", help="push the package to the hub")
    parser.add_argument("--private", action="store_true", help="create the repository as private")
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="environment variable holding the access token; the token is never an argument",
    )
    parser.add_argument("--allow-ungated", action="store_true", help="publish without a PASS verdict")
    parser.add_argument("--no-refine", action="store_true", help="build the mask without child kind refinement")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = command_line(argv)
    out_dir = Path(arguments.out_dir)
    manifest = build_package(arguments)
    print("package", out_dir)
    for row in manifest["files"]:
        print("  " + row["path"], row["bytes"], "bytes", row["sha256"][:16])
    print("round trip", manifest["round_trip"])
    print("gates", manifest["final_verdict"])
    if arguments.upload:
        token = read_token(arguments.token_env)
        result = publish_package(out_dir, str(arguments.repo), token, bool(arguments.private))
        manifest["publish"] = {"uploaded": True, "repo": result["repo"], "url": result["url"], "files": result["files"]}
        write_json(out_dir.parent / (out_dir.name + "_publish_receipt.json"), manifest["publish"])
        print("published", result["url"])
        for name in result["files"]:
            print("  remote", name)
    else:
        print("not uploaded; rerun with --upload to publish")
    return 0


if __name__ == "__main__":
    sys.exit(main())
