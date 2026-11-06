"""Ask the fine-tuned model a question and check what it answers.

Two directions, both in the exact format the model was trained on:

    phrase to logic       the everyday sentence goes in, the four formal renderings come out
    logic to phrase       a bracket tree goes in, one natural sentence comes out

After a formalization the script parses the bracket tree back, validates the operator
arities against dataset/templates/operators.json and asks the solver whether the
conditions can hold together. That turns a plausible looking answer into a checked one.

    python dataset/tools/ask_model.py --model Qwen/Qwen2.5-3B-Instruct --adapter dataset/checkpoints/sft ^
        --lang it --ask "Accendi la lampada della cucina solo se il livello di CO2 supera 50 percento"
    python dataset/tools/ask_model.py --model ... --adapter ... --verbalize "( IF [ flag 1 ] [ dim_light ] )" --lang it
    python dataset/tools/ask_model.py --model ... --adapter ... --lang de          # interactive loop

The system prompt is read from the training files, so the prompt the script builds is
byte identical to the one the model saw during training.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
DATASET_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import logic_map  # noqa: E402

LANGUAGE_NAMES = {"it": "Italian", "en": "English", "fr": "French", "es": "Spanish", "de": "German"}
NAME_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def load_operators() -> Dict[str, Dict[str, Any]]:
    path = DATASET_ROOT / "templates" / "operators.json"
    return json.loads(path.read_text(encoding="utf-8"))["operators"]


def load_system_prompts() -> Dict[str, str]:
    prompts: Dict[str, str] = {}
    for name in ("sft/train.jsonl", "sft/valid.jsonl", "sft/test.jsonl", "sft/fewshot_seed.jsonl"):
        path = DATASET_ROOT / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                task = row.get("task", "")
                if task and task not in prompts and row.get("messages"):
                    first = row["messages"][0]
                    if first.get("role") == "system":
                        prompts[task] = first["content"]
        if "formalize" in prompts and "verbalize" in prompts:
            break
    if "formalize" not in prompts:
        raise SystemExit("ERROR: no system prompt found in the sft files")
    return prompts


def parse_bracket_ast(text: str) -> Dict[str, Any]:
    """Read the balanced word back into a logic tree.

    Grammar: expression is either ( NAME child child ... ) or [ token token ... ].
    """
    source = text.strip()
    position = 0

    def skip_space() -> None:
        nonlocal position
        while position < len(source) and source[position].isspace():
            position += 1

    def parse_expression() -> Dict[str, Any]:
        nonlocal position
        skip_space()
        if position >= len(source):
            raise ValueError("unexpected end of input")
        if source[position] == "[":
            end = source.find("]", position)
            if end < 0:
                raise ValueError("leaf opened at " + str(position) + " is never closed")
            tokens = source[position + 1 : end].split()
            position = end + 1
            return {"leaf": tokens}
        if source[position] != "(":
            raise ValueError("expected ( or [ at " + str(position))
        position += 1
        skip_space()
        match = NAME_PATTERN.match(source, position)
        if not match:
            raise ValueError("expected an operator name at " + str(position))
        name = match.group(0)
        position = match.end()
        children: List[Dict[str, Any]] = []
        while True:
            skip_space()
            if position >= len(source):
                raise ValueError("node " + name + " is never closed")
            if source[position] == ")":
                position += 1
                break
            children.append(parse_expression())
        return {"op": name, "args": children}

    node = parse_expression()
    skip_space()
    if position != len(source):
        raise ValueError("extra text after the tree at " + str(position))
    return node


def describe_node(node: Dict[str, Any], registry: Dict[str, Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    try:
        logic_map.check_tree(node, registry)
        lines.append("operator check: every operator is known and has the right number of arguments")
    except Exception as error:  # registry problems are reported, not raised
        lines.append("operator check: FAILED " + str(error))
    lines.append("bracket tree: " + logic_map.bracket_ast(node))
    lines.append("skeleton: " + logic_map.bracket_skeleton(node))
    lines.append("pattern: " + logic_map.semantic_pattern(node))
    lines.append("grammar abstract: " + logic_map.gf_abstract(node))
    lines.append("smt assertion: " + logic_map.z3_assertion(node))
    return lines


def solve(node: Dict[str, Any], registry: Dict[str, Dict[str, Any]]) -> str:
    try:
        import z3
    except ImportError:
        return "solver: z3 is not installed, check skipped"
    try:
        assertion = logic_map.build_z3(node, z3, registry, {})
    except Exception as error:
        return "solver: the form could not be built (" + str(error) + ")"
    solver = z3.Solver()
    solver.add(assertion)
    result = solver.check()
    if result == z3.sat:
        return "solver: consistent, these conditions can all hold together"
    if result == z3.unsat:
        return "solver: contradiction, these conditions cannot hold together"
    return "solver: inconclusive"


def extract_payload(answer: str) -> Dict[str, str]:
    start = answer.find("{")
    end = answer.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(answer[start : end + 1])
    except ValueError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {key: str(value) for key, value in parsed.items()}


def build_messages(task: str, language: str, body: str, prompts: Dict[str, str]) -> List[Dict[str, str]]:
    language_name = LANGUAGE_NAMES.get(language, language)
    if task == "verbalize":
        user = language_name + " sentence for: " + body
        system = prompts.get("verbalize", prompts["formalize"])
    else:
        user = language_name + ": " + body
        system = prompts["formalize"]
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def load_model(args: argparse.Namespace) -> Tuple[Any, Any, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model.to(device)
    model.eval()
    print("model: " + args.model + " | adapter: " + (args.adapter or "none") + " | device: " + device)
    return tokenizer, model, device


def generate(messages: List[Dict[str, str]], tokenizer: Any, model: Any, device: str, max_new_tokens: int) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def report_formalization(answer: str, registry: Dict[str, Dict[str, Any]], with_solver: bool) -> None:
    payload = extract_payload(answer)
    print("--- answer ---")
    print(answer)
    tree = payload.get("bracket_ast", "")
    if not tree:
        print("note: the answer carries no bracket tree, nothing to check")
        return
    skeleton = re.sub(r"[^\[\]()]", "", tree)
    print("--- checks ---")
    print("brackets balanced: " + str(logic_map.is_well_bracketed(skeleton)))
    try:
        node = parse_bracket_ast(tree)
    except ValueError as error:
        print("parse: FAILED " + str(error))
        return
    for line in describe_node(node, registry):
        print(line)
    if with_solver:
        print(solve(node, registry))


def interactive(tokenizer: Any, model: Any, device: str, args: argparse.Namespace, prompts: Dict[str, str]) -> int:
    registry = load_operators()
    print("type a sentence, or quit to leave")
    while True:
        try:
            text = input("phrase> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0
        if text.lower() in ("quit", "exit", ""):
            return 0
        messages = build_messages("formalize", args.lang, text, prompts)
        answer = generate(messages, tokenizer, model, device, args.max_new_tokens)
        report_formalization(answer, registry, not args.no_solver)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Query the fine-tuned model and verify its answer.")
    parser.add_argument("--model", required=True, help="base chat model")
    parser.add_argument("--adapter", default="", help="trained adapter folder")
    parser.add_argument("--lang", default="it", help="it, en, fr, es, de")
    parser.add_argument("--ask", default="", help="everyday sentence to formalize")
    parser.add_argument("--verbalize", default="", help="bracket tree to render as a sentence")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--no-solver", action="store_true", help="skip the Z3 consistency check")
    parser.add_argument("--cpu", action="store_true", help="force the CPU even when a GPU is present")
    args = parser.parse_args(argv)

    prompts = load_system_prompts()
    tokenizer, model, device = load_model(args)

    if args.verbalize:
        registry = load_operators()
        try:
            node = parse_bracket_ast(args.verbalize)
        except ValueError as error:
            print("ERROR: the bracket tree cannot be read: " + str(error))
            return 2
        for line in describe_node(node, registry):
            print(line)
        messages = build_messages("verbalize", args.lang, logic_map.semantic_pattern(node), prompts)
        print("--- answer ---")
        print(generate(messages, tokenizer, model, device, args.max_new_tokens))
        return 0

    if args.ask:
        registry = load_operators()
        messages = build_messages("formalize", args.lang, args.ask, prompts)
        answer = generate(messages, tokenizer, model, device, args.max_new_tokens)
        report_formalization(answer, registry, not args.no_solver)
        return 0

    return interactive(tokenizer, model, device, args, prompts)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
