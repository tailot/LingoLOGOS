"""Validate the generated LingoLOGOS synthetic dataset.

The validator re-derives every formal field from the canonical AST stored in
each item, so a regenerated file that drifted from the templates is caught.
Structural checks (bracket algebra, schema, parallel groups, corpus integrity)
run first, then the formal pass builds real Z3 assertions and asks whether
each logical form is contingent: satisfiable, and still satisfiable negated.

Usage:
    python datasets/multilingual/tools/validate_dataset.py
    python datasets/multilingual/tools/validate_dataset.py --no-z3 --max-z3-forms 400
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from corpus_paths import bootstrap  # noqa: E402

REPO_ROOT, DATASET_ROOT = bootstrap(HERE)

import logic_map  # noqa: E402  (bootstrap above)

LANGUAGES = ("it", "en", "fr", "es", "de")
SPLITS = ("train", "valid", "test")
REQUIRED_FIELDS = (
    "id",
    "parallel_group",
    "template_id",
    "domain",
    "logic_family",
    "pattern_label",
    "language",
    "phrase",
    "bracket_ast",
    "bracket_skeleton",
    "skeleton_pairs",
    "ast_json",
    "ast_key",
    "semantic_pattern",
    "gf_abstract",
    "z3_assertion",
    "smtlib",
    "slot_values",
)
SFT_KEYS = ("id", "parallel_group", "template_id", "domain", "language", "task", "messages")
SFT_ROLES = ("system", "user", "assistant")
FORMALIZE_KEYS = {"bracket_ast", "semantic_pattern", "gf_abstract", "z3_assertion"}


class Report:
    """Collect failures and notes, then print one summary."""

    def __init__(self) -> None:
        self.failures: List[str] = []
        self.notes: List[str] = []

    def fail(self, message: str) -> None:
        if len(self.failures) < 40:
            self.failures.append(message)

    def note(self, message: str) -> None:
        self.notes.append(message)

    def verdict(self) -> int:
        for line in self.notes:
            print("note: " + line)
        if self.failures:
            for line in self.failures:
                print("FAIL: " + line)
            print("VERDICT: FAIL (" + str(len(self.failures)) + " problems)")
            return 1
        print("VERDICT: PASS")
        return 0


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SystemExit(str(path) + " line " + str(number) + ": " + str(error))
    return rows


def read_tree(item: Dict[str, Any]) -> Any:
    raw = item["ast_json"]
    return json.loads(raw) if isinstance(raw, str) else raw


def load_items(root: Path, report: Report) -> Dict[str, Dict[str, Any]]:
    items: Dict[str, Dict[str, Any]] = {}
    for split in SPLITS:
        files = sorted((root / "data").glob("*/" + split + "-*.jsonl"))
        if not files:
            report.fail("no data files for split " + split)
        for path in files:
            language = path.parent.name
            for item in load_jsonl(path):
                if item.get("language") != language:
                    report.fail(
                        str(path) + ": item " + str(item.get("id")) + " claims language "
                        + str(item.get("language")) + " but sits in " + language
                    )
                identifier = item.get("id")
                if identifier in items:
                    report.fail("duplicate id " + str(identifier))
                item["_split"] = split
                items[str(identifier)] = item
    if not items:
        raise SystemExit("no items found: run generate_dataset.py first")
    return items


def check_schema(items: Dict[str, Dict[str, Any]], report: Report) -> None:
    for identifier, item in items.items():
        missing = [key for key in REQUIRED_FIELDS if key not in item]
        if missing:
            report.fail(identifier + ": missing fields " + ",".join(missing))
            continue
        if item["language"] not in LANGUAGES:
            report.fail(identifier + ": unknown language " + str(item["language"]))
        if not isinstance(item["phrase"], str) or not item["phrase"].strip():
            report.fail(identifier + ": empty phrase")
        if not isinstance(item["slot_values"], dict) or not item["slot_values"]:
            report.fail(identifier + ": slot_values must be a non-empty object")


def check_brackets(items: Dict[str, Dict[str, Any]], registry: Dict[str, Any], report: Report) -> Dict[str, Any]:
    forms: Dict[str, Any] = {}
    for identifier, item in items.items():
        try:
            tree = read_tree(item)
        except json.JSONDecodeError as error:
            report.fail(identifier + ": ast_json is not valid JSON (" + str(error) + ")")
            continue
        try:
            logic_map.check_tree(tree, registry)
        except logic_map.LogicTreeError as error:
            report.fail(identifier + ": invalid AST (" + str(error) + ")")
            continue
        ast = logic_map.bracket_ast(tree)
        if ast != item["bracket_ast"]:
            report.fail(identifier + ": bracket_ast drifted from AST")
        skeleton = logic_map.bracket_skeleton(tree)
        if skeleton != item["bracket_skeleton"]:
            report.fail(identifier + ": bracket_skeleton drifted from AST")
        if not logic_map.is_well_bracketed(skeleton):
            report.fail(identifier + ": skeleton is not a balanced Dyck-2 word")
        if item["skeleton_pairs"] != len(skeleton) // 2:
            report.fail(identifier + ": skeleton_pairs disagrees with the skeleton")
        if logic_map.node_key(tree) != item["ast_key"]:
            report.fail(identifier + ": ast_key disagrees with the AST")
        for key, function in (
            ("semantic_pattern", logic_map.semantic_pattern),
            ("gf_abstract", logic_map.gf_abstract),
            ("z3_assertion", logic_map.z3_assertion),
            ("smtlib", logic_map.smtlib_text),
        ):
            if item[key] != function(tree):
                report.fail(identifier + ": " + key + " drifted from the AST")
        if item["smtlib"].count("(") != item["smtlib"].count(")"):
            report.fail(identifier + ": smtlib text has unbalanced parentheses")
        forms.setdefault(item["ast_key"], tree)
    return forms


def check_groups(items: Dict[str, Dict[str, Any]], report: Report) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items.values():
        groups[item["parallel_group"]].append(item)
    for group, members in groups.items():
        languages = sorted(member["language"] for member in members)
        if languages != sorted(LANGUAGES):
            report.fail("group " + group + " covers " + ",".join(languages) + " instead of the five languages")
        if len({member["ast_key"] for member in members}) != 1:
            report.fail("group " + group + " mixes two logical forms")
        if len({member["phrase"] for member in members}) != len(members):
            report.fail("group " + group + " repeats a surface phrase")
        if len({member["domain"] for member in members}) != 1:
            report.fail("group " + group + " mixes two domains")
    return groups


def check_corpus(root: Path, items: Dict[str, Dict[str, Any]], report: Report) -> None:
    forms_by_split = {split: set() for split in SPLITS}
    words_by_split = {split: set() for split in SPLITS}
    for item in items.values():
        words_by_split[item["_split"]].add(item["bracket_skeleton"])
        forms_by_split[item["_split"]].add(item["ast_key"])
    every = set().union(*words_by_split.values())
    all_path = root / "corpus" / "dyck2_all.txt"
    if not all_path.exists():
        report.fail("missing corpus/dyck2_all.txt")
        return
    corpus = set(all_path.read_text(encoding="utf-8").split())
    if corpus != every:
        report.fail(
            "corpus/dyck2_all.txt has " + str(len(corpus)) + " words but items define " + str(len(every))
        )
    for split in SPLITS:
        path = root / "corpus" / ("dyck2_" + split + ".txt")
        if not path.exists():
            report.fail("missing corpus file " + path.name)
            continue
        words = set(path.read_text(encoding="utf-8").split())
        if words != words_by_split[split]:
            report.fail("corpus/" + path.name + " does not match the " + split + " skeletons")
    # Logical forms must never cross splits; bracket words may legitimately repeat.
    for split, other in (("train", "valid"), ("train", "test"), ("valid", "test")):
        leak = forms_by_split[split] & forms_by_split[other]
        if leak:
            report.fail("logical form leak between " + split + " and " + other + " (" + str(len(leak)) + " forms)")
        shared = len(words_by_split[split] & words_by_split[other])
        report.note(
            "shared bracket words " + split + "/" + other + ": " + str(shared)
        )


def check_negative_controls(root: Path, items: Dict[str, Dict[str, Any]], report: Report) -> None:
    path = root / "corpus" / "dyck2_negative_controls.txt"
    if not path.exists():
        report.fail("missing corpus/dyck2_negative_controls.txt")
        return
    words = path.read_text(encoding="utf-8").split()
    positives = {item["bracket_skeleton"] for item in items.values()}
    broken = 0
    for word in words:
        if logic_map.is_well_bracketed(word):
            report.fail("negative control is still well bracketed: " + word)
        else:
            broken += 1
        if word in positives:
            report.fail("negative control collides with a positive skeleton: " + word)
    report.note("negative controls: " + str(broken) + " of " + str(len(words)) + " are ill-bracketed")


def check_sft(root: Path, items: Dict[str, Dict[str, Any]], report: Report) -> None:
    index = {(item["parallel_group"], item["language"]): item for item in items.values()}
    total = 0
    tasks: Counter = Counter()
    for split in SPLITS:
        path = root / "sft" / (split + ".jsonl")
        if not path.exists():
            report.fail("missing sft/" + path.name)
            continue
        for row in load_jsonl(path):
            total += 1
            missing = [key for key in SFT_KEYS if key not in row]
            if missing:
                report.fail(str(row.get("id")) + ": sft row misses " + ",".join(missing))
                continue
            messages = row["messages"]
            if [message["role"] for message in messages] != list(SFT_ROLES):
                report.fail(str(row["id"]) + ": sft roles must be system, user, assistant")
                continue
            if not all(message["content"].strip() for message in messages):
                report.fail(str(row["id"]) + ": empty message content")
            tasks[row["task"]] += 1
            source = index.get((row["parallel_group"], row["language"]))
            if source is None:
                report.fail(str(row["id"]) + ": no matching dataset item")
                continue
            if row["task"] == "formalize":
                try:
                    payload = json.loads(messages[2]["content"])
                except json.JSONDecodeError:
                    report.fail(str(row["id"]) + ": assistant payload is not JSON")
                    continue
                if set(payload) != FORMALIZE_KEYS:
                    report.fail(str(row["id"]) + ": assistant keys " + ",".join(sorted(payload)))
                elif payload["bracket_ast"] != source["bracket_ast"]:
                    report.fail(str(row["id"]) + ": assistant bracket_ast does not match the item")
            elif row["task"] == "verbalize":
                if messages[2]["content"] != source["phrase"]:
                    report.fail(str(row["id"]) + ": verbalized sentence does not match the item")
                if source["semantic_pattern"] not in messages[1]["content"]:
                    report.fail(str(row["id"]) + ": verbalize prompt lacks the semantic pattern")
            else:
                report.fail(str(row["id"]) + ": unknown sft task " + str(row["task"]))
    report.note(
        "sft rows: " + str(total) + " (" + ", ".join(key + "=" + str(tasks[key]) for key in sorted(tasks)) + ")"
    )


def check_configs(root: Path, report: Report) -> None:
    for name in ("all",) + LANGUAGES:
        path = root / "configs" / (name + ".json")
        if not path.exists():
            report.fail("missing configs/" + name + ".json")
            continue
        config = json.loads(path.read_text(encoding="utf-8"))
        for key in ("name", "version", "languages", "data_files"):
            if key not in config:
                report.fail("configs/" + name + ".json has no " + key)
        for split, pattern in config.get("data_files", {}).items():
            if not list(root.glob(pattern)):
                report.fail("configs/" + name + ".json pattern " + pattern + " matches nothing")


def check_distribution(items: Dict[str, Dict[str, Any]], groups: Dict[str, List[Dict[str, Any]]], report: Report) -> None:
    templates = Counter(item["template_id"] for item in items.values())
    report.note("templates in use: " + str(len(templates)))
    report.note(
        "forms per template: min " + str(min(templates.values())) + ", max " + str(max(templates.values()))
    )
    families = Counter(item["logic_family"] for item in items.values())
    report.note("logic families: " + ", ".join(str(key) + "=" + str(families[key]) for key in sorted(families)))
    domains = Counter(item["domain"] for item in items.values())
    report.note("domains: " + ", ".join(str(key) + "=" + str(domains[key]) for key in sorted(domains)))
    report.note("parallel groups: " + str(len(groups)))


def check_z3(
    forms: Dict[str, Any],
    registry: Dict[str, Any],
    max_forms: int,
    report: Report,
) -> None:
    try:
        import z3
    except ImportError:
        report.note("z3-solver is not installed: formal pass skipped")
        return
    keys = sorted(forms)
    if max_forms > 0:
        keys = keys[:max_forms]
    solver = z3.Solver()
    refuter = z3.Solver()
    unsat: List[str] = []
    tautologies: List[str] = []
    errors: List[str] = []
    started = time.time()
    for key in keys:
        tree = forms[key]
        try:
            assertion = logic_map.build_z3(tree, z3, registry, {})
        except Exception as error:  # a broken form must fail the run, not the validator
            errors.append(key + ": " + type(error).__name__ + " " + str(error))
            continue
        solver.push()
        solver.add(assertion)
        first = solver.check()
        refuter.push()
        refuter.add(z3.Not(assertion))
        second = refuter.check()
        solver.pop()
        refuter.pop()
        if first != z3.sat:
            unsat.append(key)
        elif second != z3.sat:
            tautologies.append(key)
    elapsed = time.time() - started
    report.note(
        "formal pass: " + str(len(keys)) + " logical forms in " + format(elapsed, ".1f") + " s"
        + ", contingent " + str(len(keys) - len(unsat) - len(tautologies) - len(errors))
    )
    for key in unsat[:5]:
        report.fail("form " + key + " is never satisfiable (contradiction in the template)")
    for key in tautologies[:5]:
        report.fail("form " + key + " is valid in every model (its negation is unsatisfiable)")
    for message in errors[:5]:
        report.fail("Z3 build failed for " + message)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Validate the LingoLOGOS synthetic dataset.")
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT), help="dataset directory to check")
    parser.add_argument("--no-z3", action="store_true", help="skip the formal Z3 pass")
    parser.add_argument("--max-z3-forms", type=int, default=0, help="cap the number of forms sent to Z3")
    args = parser.parse_args(argv)

    root = Path(args.dataset_root).resolve()
    registry = json.loads((root / "templates" / "operators.json").read_text(encoding="utf-8"))["operators"]
    report = Report()

    items = load_items(root, report)
    report.note("items: " + str(len(items)))
    check_schema(items, report)
    forms = check_brackets(items, registry, report)
    report.note("distinct logical forms: " + str(len(forms)))
    groups = check_groups(items, report)
    check_corpus(root, items, report)
    check_negative_controls(root, items, report)
    check_sft(root, items, report)
    check_configs(root, report)
    check_distribution(items, groups, report)
    if not args.no_z3:
        check_z3(forms, registry, args.max_z3_forms, report)
    return report.verdict()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
