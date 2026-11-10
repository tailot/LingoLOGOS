"""Generate the LingoLOGOS multilingual synthetic dataset.

The script expands the template bank in datasets/multilingual/templates into a deterministic
corpus of everyday recurrence instructions in Italian, English, French, Spanish
and German. Every logical form is emitted once per language inside the same
parallel group, together with its bracket algebra (a word of Dyck-2 over the
terminals the project engine induces), its functional semantic pattern, its
Grammatical Framework abstract syntax and its Z3 assertion.

Outputs, all under the dataset repository:
  data/<lang>/{train,valid,test}-NNNNN.jsonl   items per language and split
  corpus/dyck2_{train,valid,test}.txt         bracket skeletons for grammar induction
  corpus/dyck2_negative_controls.txt          malformed skeletons for rejection gates
  sft/{train,valid,test}.jsonl                chat pairs for instruction tuning
  sft/fewshot_seed.jsonl                      50 seed pairs, 10 per domain
  configs/*.json                              Hugging Face data_files maps
  stats.json                                  corpus statistics and audits

Run: python datasets/multilingual/tools/generate_dataset.py --combos-per-template 40
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from corpus_paths import bootstrap  # noqa: E402

REPO_ROOT, DATASET_ROOT = bootstrap(HERE)

import logic_map  # noqa: E402  (path bootstrap above)

LANGUAGES = ("it", "en", "fr", "es", "de")
DOMAINS = ("domotica", "agenda", "finanze", "salute", "produttivita")
SPLITS = ("train", "valid", "test")
NUMERIC_CLASSES = {
    "value_pct",
    "value_pct_low",
    "value_temp",
    "value_temp_low",
    "value_pct_pair",
    "value_temp_pair",
    "money_amount",
    "balance_level",
    "money_pair",
    "pct_point",
    "count_days_short",
    "count_days_long",
    "count_hours",
    "count_minutes",
    "count_small",
    "storage_level",
    "value_kwh",
    "cycle_monthly",
}
REF_PATTERN = re.compile(r"\{([a-z_][a-z0-9_]*)(?:\.([a-z_]+))?(?:\|([a-z_]+))?\}")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_templates(templates_dir: Path) -> List[Dict[str, Any]]:
    bank: List[Dict[str, Any]] = []
    for domain in DOMAINS:
        payload = load_json(templates_dir / (domain + ".json"))
        if payload.get("domain") != domain:
            raise SystemExit("template file " + domain + ".json declares a mismatched domain")
        for template in payload["templates"]:
            template["domain"] = domain
            bank.append(template)
    return bank


def entry_id(entry: Dict[str, Any]) -> str:
    return str(entry["id"])


def parse_ref(token: str) -> Tuple[str, str, str]:
    match = REF_PATTERN.fullmatch(token)
    if not match:
        raise SystemExit("malformed slot reference " + token)
    return match.group(1), match.group(2) or "", match.group(3) or ""


def surface(
    lexicon: Dict[str, Any],
    slot_class: str,
    entry: Dict[str, Any],
    language: str,
    field: str = "",
    variant: str = "",
) -> str:
    """Surface form of a slot value in one language."""
    if field:
        fmt = lexicon["number_formats"][entry.get("unit", "")][language]
        return fmt.format(v=entry[field])
    if slot_class == "clock_time":
        return lexicon["clock_formats"][language].format(t=entry["value"])
    if slot_class == "cycle_monthly":
        return entry[language]
    if "value" in entry and "unit" in entry:
        fmt = lexicon["number_formats"][entry["unit"]][language]
        return fmt.format(v=entry["value"])
    if variant:
        flat = entry.get(language + "_" + variant)
        if isinstance(flat, str):
            return flat
        nested = entry.get(variant)
        if isinstance(nested, dict) and language in nested:
            return nested[language]
    return entry[language]


def leaf_for_entry(slot_class: str, entry: Dict[str, Any], field: str = "") -> List[str]:
    """Canonical algebra leaf tokens of a slot value."""
    if slot_class == "clock_time":
        hours, minutes = str(entry["value"]).split(":")
        return ["clock", hours, minutes]
    if slot_class in ("ordinal_k", "ordinal_last"):
        return ["ord", str(entry["ord"])]
    if slot_class == "time_band":
        return ["band", entry_id(entry)]
    if "value" in entry and "unit" in entry:
        value = entry[field] if field else entry["value"]
        return [entry["unit"], str(value)]
    return [entry_id(entry)]


def build_ast(
    spec: Any,
    lexicon: Dict[str, Any],
    slots: Dict[str, str],
    assignment: Dict[str, Dict[str, Any]],
    derived: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Turn a template AST specification into a resolved logic tree."""
    if isinstance(spec, list):
        head = spec[0]
        if not isinstance(head, str) or not head.isupper():
            raise SystemExit("template node must start with an operator name: " + repr(spec))
        return {
            "op": head,
            "args": [build_ast(child, lexicon, slots, assignment, derived) for child in spec[1:]],
        }
    if isinstance(spec, str):
        if spec.startswith("{"):
            name, field, _variant = parse_ref(spec)
            pool = dict(assignment)
            pool.update(derived)
            if name not in pool:
                raise SystemExit("unknown slot reference " + name)
            if name in assignment:
                slot_class = slots[name]
            else:
                slot_class = "derived"
            return {"leaf": leaf_for_entry(slot_class, pool[name], field)}
        if spec.isupper():
            return {"op": spec, "args": []}
        return {"leaf": [spec]}
    raise SystemExit("unsupported AST element " + repr(spec))


def render_phrase(
    template_text: str,
    lexicon: Dict[str, Any],
    slots: Dict[str, str],
    assignment: Dict[str, Dict[str, Any]],
    derived: Dict[str, Dict[str, Any]],
    language: str,
) -> str:
    """Fill a natural language template with the surface forms of one language."""

    def replace(match: "re.Match[str]") -> str:
        name = match.group(1)
        field = match.group(2) or ""
        variant = match.group(3) or ""
        if name in assignment:
            return surface(lexicon, slots[name], assignment[name], language, field, variant)
        if name in derived:
            entry = derived[name]
            fmt = lexicon["number_formats"][entry["unit"]][language]
            return fmt.format(v=entry["value"])
        raise SystemExit("phrase references unknown slot " + name)

    text = REF_PATTERN.sub(replace, template_text).strip()
    while "  " in text:
        text = text.replace("  ", " ")
    if text and text[0].isalpha():
        text = text[0].upper() + text[1:]
    return text


def distinct_within_class(names: Sequence[str], classes: Dict[str, str], picks: Sequence[Dict[str, Any]]) -> bool:
    """Two slots bound to the same class must not receive the same entry."""
    seen: Dict[Tuple[str, str], int] = {}
    for name, entry in zip(names, picks):
        key = (classes[name], entry_id(entry))
        if key in seen:
            return False
        seen[key] = 1
    return True


def sample_combos(
    template: Dict[str, Any],
    lexicon: Dict[str, Any],
    needed: int,
    rng: Any,
) -> List[Dict[str, Dict[str, Any]]]:
    """Pick distinct slot assignments for one template, deterministically."""
    slots = template["slots"]
    names = sorted(slots)
    classes = {name: slots[name] for name in names}
    pools = {name: lexicon["slots"][classes[name]] for name in names}
    product_size = 1
    for name in names:
        product_size *= len(pools[name])

    combos: List[Dict[str, Dict[str, Any]]] = []
    seen_keys = set()

    def register(picks: Sequence[Dict[str, Any]]) -> bool:
        if not distinct_within_class(names, classes, picks):
            return False
        assignment = {name: entry for name, entry in zip(names, picks)}
        derived = derive_values(template, assignment, rng)
        key = tuple((classes[name], entry_id(entry)) for name, entry in zip(names, picks))
        key += tuple((name, str(entry["value"])) for name, entry in sorted(derived.items()))
        if key in seen_keys:
            return False
        seen_keys.add(key)
        combos.append({"assignment": assignment, "derived": derived})
        return True

    if product_size <= needed * 4:
        for picks in itertools.product(*(pools[name] for name in names)):
            if len(combos) >= needed:
                break
            register(picks)
        if len(combos) >= needed:
            return combos

    attempts = 0
    limit = max(needed * 400, 4000)
    while len(combos) < needed and attempts < limit:
        attempts += 1
        register([rng.choice(pools[name]) for name in names])
    if len(combos) < needed:
        print("note: template " + template["id"] + " supports " + str(len(combos)) + " distinct forms, requested " + str(needed))
    return combos


def derive_values(template: Dict[str, Any], assignment: Dict[str, Dict[str, Any]], rng: Any) -> Dict[str, Dict[str, Any]]:
    """Build derived slots, for example the threshold that belongs to a health indicator."""
    derived: Dict[str, Dict[str, Any]] = {}
    for name, rule in template.get("derived", {}).items():
        source = assignment[rule["from"]]
        values = source[rule["field"]]
        chosen = values[rng.randrange(len(values))]
        derived[name] = {"id": rule["from"] + "_" + str(chosen), "value": chosen, "unit": source.get("unit", "count")}
    return derived


def make_item(
    template: Dict[str, Any],
    group: str,
    combo: Dict[str, Any],
    lexicon: Dict[str, Any],
    registry: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Render one logical form as five parallel items, one per language."""
    slots = template["slots"]
    assignment = combo["assignment"]
    derived = combo["derived"]
    tree = build_ast(template["ast"], lexicon, slots, assignment, derived)
    logic_map.check_tree(tree, registry)

    ast_text = logic_map.bracket_ast(tree)
    skeleton = logic_map.bracket_skeleton(tree)
    if not logic_map.is_well_bracketed(skeleton):
        raise SystemExit("template " + template["id"] + " produced an unbalanced skeleton")
    semantic = logic_map.semantic_pattern(tree)
    gf_text = logic_map.gf_abstract(tree)
    z3_text = logic_map.z3_assertion(tree)
    smt_text = logic_map.smtlib_text(tree)
    key = logic_map.node_key(tree)

    slot_map = {name: entry_id(entry) for name, entry in sorted(assignment.items())}
    for name, entry in sorted(derived.items()):
        slot_map[name] = str(entry["id"])

    items: List[Dict[str, Any]] = []
    for language in LANGUAGES:
        phrase = render_phrase(template["phrases"][language], lexicon, slots, assignment, derived, language)
        items.append(
            {
                "id": group + "-" + language,
                "parallel_group": group,
                "template_id": template["id"],
                "domain": template["domain"],
                "logic_family": template["family"],
                "pattern_label": template.get("pattern_label", ""),
                "language": language,
                "phrase": phrase,
                "bracket_ast": ast_text,
                "bracket_skeleton": skeleton,
                "skeleton_pairs": len(skeleton) // 2,
                "ast_json": json.loads(json.dumps(tree)),
                "ast_key": key,
                "semantic_pattern": semantic,
                "gf_abstract": gf_text,
                "z3_assertion": z3_text,
                "smtlib": smt_text,
                "slot_values": slot_map,
            }
        )
    return items


def generate(
    combos_per_template: int,
    seed: int,
    out_root: Path,
    train_ratio: float,
    valid_ratio: float,
    shard_lines: int,
) -> Dict[str, Any]:
    import random

    templates_dir = DATASET_ROOT / "templates"
    lexicon = load_json(templates_dir / "lexicon.json")
    registry = load_json(templates_dir / "operators.json")["operators"]
    templates = load_templates(templates_dir)

    groups: List[Dict[str, Any]] = []
    seen_forms = set()
    for template in templates:
        rng = random.Random(str(seed) + ":" + template["id"])
        combos = sample_combos(template, lexicon, combos_per_template, rng)
        accepted: List[List[Dict[str, Any]]] = []
        for combo in combos:
            items = make_item(template, template["id"], combo, lexicon, registry)
            form_keys = {item["ast_key"] for item in items}
            if form_keys & seen_forms:
                continue
            seen_forms |= form_keys
            accepted.append(items)
        for position, items in enumerate(accepted):
            group_id = template["id"] + "-" + str(position).zfill(4)
            for item in items:
                item["parallel_group"] = group_id
                item["id"] = group_id + "-" + item["language"]
            groups.append({"group": group_id, "items": items})

    # Split by parallel group so that every language of a logical form lands in one split.
    ordered = sorted(groups, key=lambda entry: entry["group"])
    rng_split = random.Random(str(seed) + ":split")
    rng_split.shuffle(ordered)
    total = len(ordered)
    n_train = int(round(total * train_ratio))
    n_valid = int(round(total * valid_ratio))
    split_of = {}
    for position, entry in enumerate(ordered):
        if position < n_train:
            split_of[entry["group"]] = "train"
        elif position < n_train + n_valid:
            split_of[entry["group"]] = "valid"
        else:
            split_of[entry["group"]] = "test"

    rows = {split: [] for split in SPLITS}
    for entry in ordered:
        split = split_of[entry["group"]]
        rows[split].extend(entry["items"])

    for language in LANGUAGES:
        for split in SPLITS:
            directory = out_root / "data" / language
            directory.mkdir(parents=True, exist_ok=True)
            for stale in directory.glob(split + "-*.jsonl"):
                stale.unlink()
            selected = [row for row in rows[split] if row["language"] == language]
            selected.sort(key=lambda row: row["id"])
            for shard_index, start in enumerate(range(0, len(selected), shard_lines)):
                chunk = selected[start : start + shard_lines]
                path = directory / (split + "-" + str(shard_index).zfill(5) + ".jsonl")
                write_jsonl(path, chunk)

    write_corpus(out_root, rows)
    write_sft(out_root, rows)
    write_configs(out_root)
    stats = build_stats(rows, groups)
    (out_root / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_corpus(out_root: Path, rows: Dict[str, List[Dict[str, Any]]]) -> None:
    import random

    corpus_dir = out_root / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261106)
    for split in SPLITS:
        skeletons = sorted({row["bracket_skeleton"] for row in rows[split]}, key=lambda text: (len(text), text))
        (corpus_dir / ("dyck2_" + split + ".txt")).write_text("\n".join(skeletons) + "\n", encoding="ascii")
    all_skeletons = sorted(
        {row["bracket_skeleton"] for split in SPLITS for row in rows[split]},
        key=lambda text: (len(text), text),
    )
    (corpus_dir / "dyck2_all.txt").write_text("\n".join(all_skeletons) + "\n", encoding="ascii")
    controls = []
    for skeleton in all_skeletons:
        broken = logic_map.negative_control(skeleton, rng)
        if broken != skeleton and not logic_map.is_well_bracketed(broken):
            controls.append(broken)
    controls = sorted(set(controls), key=lambda text: (len(text), text))
    (corpus_dir / "dyck2_negative_controls.txt").write_text("\n".join(controls) + "\n", encoding="ascii")


SYSTEM_FORMALIZE = (
    "You are LingoLOGOS. You map an everyday recurrence instruction to its formal logic. "
    "Round brackets ( ) mark operator nodes and square brackets [ ] mark value and entity leaves, "
    "so the bracket sequence is always a balanced word. Reply with JSON holding the keys "
    "bracket_ast, semantic_pattern, gf_abstract and z3_assertion."
)
SYSTEM_VERBALIZE = (
    "You are LingoLOGOS. You read the formal logic of a recurrence rule and say it back in natural "
    "language, keeping every number, unit and cycle exactly as written. Reply with one sentence."
)


def assistant_payload(row: Dict[str, Any]) -> str:
    return json.dumps(
        {
            "bracket_ast": row["bracket_ast"],
            "semantic_pattern": row["semantic_pattern"],
            "gf_abstract": row["gf_abstract"],
            "z3_assertion": row["z3_assertion"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def sft_pairs(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    language_names = {"it": "Italian", "en": "English", "fr": "French", "es": "Spanish", "de": "German"}
    name = language_names[row["language"]]
    forward = {
        "id": row["id"] + ":formalize",
        "task": "formalize",
        "template_id": row["template_id"],
        "logic_family": row["logic_family"],
        "domain": row["domain"],
        "language": row["language"],
        "parallel_group": row["parallel_group"],
        "messages": [
            {"role": "system", "content": SYSTEM_FORMALIZE},
            {"role": "user", "content": name + ": " + row["phrase"]},
            {"role": "assistant", "content": assistant_payload(row)},
        ],
    }
    backward = {
        "id": row["id"] + ":verbalize",
        "task": "verbalize",
        "template_id": row["template_id"],
        "logic_family": row["logic_family"],
        "domain": row["domain"],
        "language": row["language"],
        "parallel_group": row["parallel_group"],
        "messages": [
            {"role": "system", "content": SYSTEM_VERBALIZE},
            {"role": "user", "content": name + " sentence for: " + row["semantic_pattern"]},
            {"role": "assistant", "content": row["phrase"]},
        ],
    }
    return [forward, backward]


def write_sft(out_root: Path, rows: Dict[str, List[Dict[str, Any]]]) -> None:
    sft_dir = out_root / "sft"
    sft_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        pairs: List[Dict[str, Any]] = []
        for row in sorted(rows[split], key=lambda item: item["id"]):
            pairs.extend(sft_pairs(row))
        write_jsonl(sft_dir / (split + ".jsonl"), pairs)

    # Seed set: ten groups per domain, spread across templates, cycling through languages.
    seed_rows: List[Dict[str, Any]] = []
    for domain in DOMAINS:
        per_template: Dict[str, List[Dict[str, Any]]] = {}
        for row in sorted(rows["train"], key=lambda item: item["id"]):
            if row["domain"] == domain:
                per_template.setdefault(row["template_id"], []).append(row)
        buckets: Dict[str, List[List[Dict[str, Any]]]] = {}
        for template_id, items in per_template.items():
            per_group: Dict[str, List[Dict[str, Any]]] = {}
            for row in items:
                per_group.setdefault(row["parallel_group"], []).append(row)
            buckets[template_id] = [sorted(value, key=lambda item: item["language"]) for value in per_group.values()]
        queues = {template_id: list(buckets[template_id]) for template_id in sorted(buckets)}
        picked: List[List[Dict[str, Any]]] = []
        while len(picked) < 10 and any(queues.values()):
            for template_id in sorted(queues):
                if queues[template_id] and len(picked) < 10:
                    picked.append(queues[template_id].pop(0))
        for position, group in enumerate(picked):
            wanted = LANGUAGES[position % len(LANGUAGES)]
            seed_rows.append(next((item for item in group if item["language"] == wanted), group[0]))
    write_jsonl(sft_dir / "fewshot_seed.jsonl", [sft_pairs(row)[0] for row in seed_rows])


def write_configs(out_root: Path) -> None:
    configs = out_root / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    for language in ("all",) + LANGUAGES:
        if language == "all":
            payload = {
                "name": "all",
                "version": "1.0.0",
                "description": "All five languages, parallel groups kept inside one split.",
                "languages": list(LANGUAGES),
                "data_files": {split: "data/*/" + split + "-*.jsonl" for split in SPLITS},
            }
        else:
            payload = {
                "name": language,
                "version": "1.0.0",
                "description": "Items of one language.",
                "languages": [language],
                "data_files": {split: "data/" + language + "/" + split + "-*.jsonl" for split in SPLITS},
            }
        (configs / (language + ".json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_stats(rows: Dict[str, List[Dict[str, Any]]], groups: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_rows = [row for split in SPLITS for row in rows[split]]
    by_domain: Dict[str, int] = {}
    by_family: Dict[str, int] = {}
    by_language: Dict[str, int] = {}
    lengths: List[int] = []
    for row in all_rows:
        by_domain[row["domain"]] = by_domain.get(row["domain"], 0) + 1
        by_family[row["logic_family"]] = by_family.get(row["logic_family"], 0) + 1
        by_language[row["language"]] = by_language.get(row["language"], 0) + 1
        lengths.append(row["skeleton_pairs"])
    skeletons = {row["bracket_skeleton"] for row in all_rows}
    return {
        "languages": list(LANGUAGES),
        "domains": list(DOMAINS),
        "items_total": len(all_rows),
        "logical_forms": len(groups),
        "splits": {split: len(rows[split]) for split in SPLITS},
        "by_domain": dict(sorted(by_domain.items())),
        "by_logic_family": dict(sorted(by_family.items())),
        "by_language": by_language,
        "unique_bracket_skeletons": len(skeletons),
        "unique_semantic_patterns": len({row["semantic_pattern"] for row in all_rows}),
        "unique_phrases": len({row["phrase"] for row in all_rows}),
        "skeleton_pairs_min": min(lengths),
        "skeleton_pairs_max": max(lengths),
        "skeleton_pairs_mean": round(sum(lengths) / len(lengths), 3),
        "operators": sorted({node["op"] for row in all_rows for node in logic_map.walk(row["ast_json"]) if logic_map.is_operator_node(node)}),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the LingoLOGOS synthetic multilingual dataset.")
    parser.add_argument("--out-root", default=str(DATASET_ROOT), help="dataset repository root")
    parser.add_argument("--combos-per-template", type=int, default=40, help="distinct logical forms per template")
    parser.add_argument("--seed", type=int, default=13, help="deterministic seed")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--shard-lines", type=int, default=2000, help="maximum lines per jsonl shard")
    parser.add_argument("--list-only", action="store_true", help="print the template inventory and exit")
    args = parser.parse_args(argv)

    if args.list_only:
        templates = load_templates(Path(args.out_root) / "templates")
        for template in templates:
            print(template["id"], template["domain"], template["family"], template.get("pattern_label", ""), len(template["slots"]))
        print("templates:", len(templates))
        return 0

    stats = generate(
        combos_per_template=args.combos_per_template,
        seed=args.seed,
        out_root=Path(args.out_root),
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        shard_lines=args.shard_lines,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
