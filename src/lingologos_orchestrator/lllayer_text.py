"""Text routing for lllayer: natural language in, validated syntax tree out.

RFC identifier: LingoLOGOS-MULTI-ORCH-v1
Interface Layer: lllayer (CLI Wrapper)

``lllayer --data TEXT`` asks one isolated node to turn a sentence into a syntax
tree. The tree always comes out of a constrained decoding session: the admissible
token set of a grammar is folded into the volatile mask vector of
:class:`~lingologos_orchestrator.lllayer_masker.NativeASTLogitsProcessor`, so a
structurally impossible token cannot be sampled at all, and the finished stream is
validated before it is printed. Three grammar sources are available and the node
stays generic when none of them is requested:

* ``generic``: the built-in node AST alphabet of the node wrapper, independent of
  any corpus or topology file. It is the same session ``--constrained-decoding``
  runs inside a turn, reached directly from the command line.
* ``topology``: the node names and the routing matrix of a declared topology
  (``--topology-config``). The sentence has to name the endpoints of a declared
  directed edge, and edges that are not in the routing matrix never enter the
  mask, so an illegal route is unrepresentable rather than filtered afterwards.
* ``dataset``: the encoder-decoder weights and the bracket algebra automaton of a
  release package (``--model-package`` or ``--dataset-root``). The trained model
  proposes logits, the automaton of ``neuro_symbolic.masking`` restricts them, and
  the decoded stream is replayed through the automaton as a second check.

The masking stacks are folded here: the admissible ids produced by the symbolic
automaton are handed to the same volatile mask vector of the C kernel, which is
also what the orchestrator programs through ``--topology-ctrl-reg``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .lllayer_engine import IncrementalASTEngine, MockRunnerModel, SimpleASTGrammar
from .lllayer_masker import NativeASTLogitsProcessor
from .schema import OrchestratorError, TopologyConfig

GRAMMAR_GENERIC = "generic"
GRAMMAR_TOPOLOGY = "topology"
GRAMMAR_DATASET = "dataset"
GRAMMAR_AUTO = "auto"
GRAMMAR_CHOICES = (GRAMMAR_AUTO, GRAMMAR_GENERIC, GRAMMAR_TOPOLOGY, GRAMMAR_DATASET)

OPEN_NODE = "("
CLOSE_NODE = ")"
TRANSITION_TOKEN = "Transition"
MODIFIER_TOKEN = "MODIFIER"

# Built-in AST alphabet of the node wrapper (historical names kept byte-identical).
GENERIC_TOKENS = ["ROOT", OPEN_NODE, TRANSITION_TOKEN, "StateA", "StateB", CLOSE_NODE, "END", MODIFIER_TOKEN]

# Words that make the optional modifier branch of the AST alphabet fire.
MODIFIER_WORDS = {
    "urgent",
    "urgently",
    "urgente",
    "urgentemente",
    "urgentement",
    "urgenza",
    "dringend",
    "immediately",
    "immediatamente",
    "subito",
}

# Files that make a directory a self-contained release package.
PACKAGE_TOKENIZER_FILE = "tokenizer.json"
PACKAGE_MASK_FILE = "mask_spec.json"
PACKAGE_WEIGHT_CANDIDATES = ("model.safetensors", "model.pt")
PACKAGE_SEARCH_CANDIDATES = (".", "checkpoints/hub", "checkpoints/release", "checkpoints")


class TextRoutingError(OrchestratorError):
    """Raised when a sentence cannot be routed into a valid tree."""


def fold_text(text: str) -> str:
    """Lower case ASCII fold used to match evidence words inside a sentence."""
    decomposed = unicodedata.normalize("NFKD", str(text))
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", ascii_only.lower()).strip()


def mention_position(word: str, folded_text: str) -> int:
    """Return the offset of the first whole word mention, or -1 when absent."""
    needle = fold_text(word)
    if not needle:
        return -1
    match = re.search(r"(?:^| )" + re.escape(needle) + r"(?: |$)", " " + folded_text + " ")
    return match.start() if match else -1


def first_mention(words: Iterable[str], folded_text: str) -> int:
    """Offset of the earliest mention among a set of words, or -1 when none occurs."""
    positions = [position for position in (mention_position(word, folded_text) for word in words) if position >= 0]
    return min(positions) if positions else -1


@dataclass
class RoutedTree:
    """Decoding result of one text query, with the evidence that justified it."""

    grammar: str
    tree: str
    skeleton: str
    symbols: List[str] = field(default_factory=list)
    steps: int = 0
    evidence: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    valid: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grammar": self.grammar,
            "tree": self.tree,
            "skeleton": self.skeleton,
            "symbols": list(self.symbols),
            "steps": self.steps,
            "valid": self.valid,
            "evidence": dict(self.evidence),
            "model": dict(self.model),
        }


def skeleton_of_parentheses(tree: str) -> str:
    """Project a decoded AST word onto its node bracket terminals."""
    return "".join(char for char in tree if char in (OPEN_NODE, CLOSE_NODE))


def validate_parentheses(tree: str) -> None:
    """Raise when a decoded AST word is not one balanced parenthesis word."""
    depth = 0
    for char in tree:
        if char == OPEN_NODE:
            depth += 1
        elif char == CLOSE_NODE:
            depth -= 1
            if depth < 0:
                raise TextRoutingError("the decoded tree closes a node that was never opened")
    if depth != 0:
        raise TextRoutingError("the decoded tree stops with " + str(depth) + " node(s) still open")


# ---------------------------------------------------------------------------
# Generic route: the built-in AST alphabet of the node wrapper
# ---------------------------------------------------------------------------
def generic_grammar() -> Tuple[SimpleASTGrammar, Dict[str, int], Dict[int, str]]:
    """Return the built-in node AST grammar of the wrapper."""
    token_to_id = {token: index for index, token in enumerate(GENERIC_TOKENS)}
    id_to_token = {index: token for index, token in enumerate(GENERIC_TOKENS)}
    grammar = SimpleASTGrammar(
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        node_types={MODIFIER_TOKEN: "OPTIONAL_MODIFIER"},
        transitions={
            ("ROOT", OPEN_NODE): "NODE_OPN",
            ("NODE_OPN", TRANSITION_TOKEN): "NODE_TRANS",
            ("NODE_TRANS", "StateA"): "NODE_S1",
            ("NODE_S1", "StateB"): "NODE_S2",
            ("NODE_S2", CLOSE_NODE): "TERMINAL",
            ("NODE_TRANS", MODIFIER_TOKEN): "MODIFIER",
            ("MODIFIER", CLOSE_NODE): "TERMINAL",
        },
        valid_tokens={
            "ROOT": {token_to_id[OPEN_NODE]},
            "NODE_OPN": {token_to_id[TRANSITION_TOKEN]},
            "NODE_TRANS": {token_to_id["StateA"], token_to_id[MODIFIER_TOKEN]},
            "NODE_S1": {token_to_id["StateB"]},
            "NODE_S2": {token_to_id[CLOSE_NODE]},
            "MODIFIER": {token_to_id[CLOSE_NODE]},
        },
        closing_tokens={token_to_id[CLOSE_NODE]},
        terminal_nodes={"TERMINAL", "END"},
    )
    return grammar, token_to_id, id_to_token


class LexicalRunnerModel(MockRunnerModel):
    """Evidence-only runner: fixed logits over one AST alphabet.

    The mask already removes every inadmissible token, so these logits only order
    what the automaton left admissible. A token scores above zero only when the
    sentence mentions it, which is what keeps a declared-but-unmentioned node id
    out of the answer.
    """

    def __init__(
        self,
        vocab_size: int,
        token_to_id: Dict[str, int],
        id_to_token: Dict[int, str],
        scores: Sequence[float],
    ):
        super().__init__(vocab_size, token_to_id, id_to_token)
        self.scores = np.asarray(scores, dtype=np.float32)

    def get_logits(self, tokens: List[int]) -> np.ndarray:
        return self.scores.copy()


def evidence_scores(
    token_to_id: Dict[str, int],
    folded_text: str,
    mentioned: Dict[int, int],
    vocab_size: int,
    structural: Iterable[str] = (OPEN_NODE, CLOSE_NODE, TRANSITION_TOKEN),
) -> List[float]:
    """Build the evidence logits of one AST alphabet from the mention offsets."""
    limit = max(1, len(folded_text))
    scores = np.zeros(vocab_size, dtype=np.float32)
    for token, token_id in token_to_id.items():
        if token_id >= vocab_size:
            continue
        if token in structural:
            scores[token_id] = 1.0
            continue
        position = mentioned.get(token_id, -1)
        if position >= 0:
            scores[token_id] = 1.0 + (limit - position) / limit
    return list(scores)


def run_engine_session(
    grammar: SimpleASTGrammar,
    runner: LexicalRunnerModel,
    prompt: str,
    max_steps: int = 128,
) -> Tuple[str, int]:
    """Run one constrained decoding session and return the tree word and its length."""
    engine = IncrementalASTEngine(
        grammar_spec=grammar,
        logits_processor=NativeASTLogitsProcessor(vocab_size=runner.vocab_size),
    )
    tree = engine.run_session(prompt=prompt, runner_model=runner, max_steps=max_steps)
    return tree, len(tree.split())


def route_generic_text(text: str) -> RoutedTree:
    """Route a sentence through the built-in AST alphabet of the node wrapper."""
    grammar, token_to_id, id_to_token = generic_grammar()
    folded = fold_text(text)
    mentioned: Dict[int, int] = {}
    modifier_position = first_mention(MODIFIER_WORDS, folded)
    if modifier_position >= 0:
        mentioned[token_to_id[MODIFIER_TOKEN]] = modifier_position
    scores = evidence_scores(token_to_id, folded, mentioned, vocab_size=len(GENERIC_TOKENS))
    runner = LexicalRunnerModel(
        vocab_size=len(GENERIC_TOKENS),
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        scores=scores,
    )
    tree, steps = run_engine_session(grammar, runner, prompt=text)
    validate_parentheses(tree)
    return RoutedTree(
        grammar=GRAMMAR_GENERIC,
        tree=tree,
        skeleton=skeleton_of_parentheses(tree),
        symbols=tree.split(),
        steps=steps,
        evidence={
            "text": text,
            "alphabet": list(GENERIC_TOKENS),
            "modifierMentioned": bool(mentioned),
            "note": "the built-in alphabet is independent of any corpus or topology file",
        },
        valid=True,
    )


# ---------------------------------------------------------------------------
# Topology route: node names and routing matrix of a declared topology
# ---------------------------------------------------------------------------
@dataclass
class TopologyBinding:
    """Node ids and directed edges a node may route a message through."""

    session_id: str
    node_ids: List[str]
    adjacency: Dict[str, List[str]]
    source_path: str = ""

    def route_exists(self, source: str, target: str) -> bool:
        return target in self.adjacency.get(source, [])

    def routes(self) -> List[str]:
        return [
            source + " -> " + target
            for source in self.node_ids
            for target in self.adjacency.get(source, [])
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sessionId": self.session_id,
            "nodes": list(self.node_ids),
            "routingMatrix": {node: list(self.adjacency.get(node, [])) for node in self.node_ids},
            "source": self.source_path,
        }


def load_topology_binding(path: Path) -> TopologyBinding:
    """Read the declared node ids and routing matrix of a topology configuration."""
    source = Path(path)
    if not source.is_file():
        raise TextRoutingError("topology configuration not found: " + str(source))
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TextRoutingError("topology configuration is not valid JSON: " + str(source)) from error
    config = TopologyConfig.from_dict(data)
    adjacency = config.get_adjacency_map()
    return TopologyBinding(
        session_id=config.sessionId,
        node_ids=[instance.id for instance in config.instances],
        adjacency=adjacency,
        source_path=str(source),
    )


def name_variants(node_id: str, all_node_ids: Sequence[str]) -> List[str]:
    """Matchable spellings of a node id: the id itself and its unique segments."""
    variants = [node_id]
    segments = [segment for segment in re.split(r"[^0-9A-Za-z]+", node_id) if segment]
    if len(segments) > 1:
        counts: Dict[str, int] = {}
        for other in all_node_ids:
            for segment in [piece for piece in re.split(r"[^0-9A-Za-z]+", other) if piece]:
                counts[segment] = counts.get(segment, 0) + 1
        variants.extend(segment for segment in segments if counts.get(segment, 0) == 1)
    return variants


def mentioned_nodes(text: str, binding: TopologyBinding) -> List[Tuple[str, int]]:
    """Declared node ids named by a sentence, in order of first mention."""
    folded = fold_text(text)
    found: List[Tuple[str, int]] = []
    for node_id in binding.node_ids:
        positions = [
            position
            for position in (mention_position(variant, folded) for variant in name_variants(node_id, binding.node_ids))
            if position >= 0
        ]
        if positions:
            found.append((node_id, min(positions)))
    found.sort(key=lambda item: (item[1], item[0]))
    return found


def topology_grammar(
    binding: TopologyBinding,
    sources: Sequence[str],
    targets_by_source: Dict[str, List[str]],
) -> Tuple[SimpleASTGrammar, Dict[str, int], Dict[int, str]]:
    """Build the AST alphabet of one routing query, with the routing matrix in the mask."""
    tokens = ["ROOT", OPEN_NODE, TRANSITION_TOKEN, CLOSE_NODE, MODIFIER_TOKEN]
    tokens.extend(binding.node_ids)
    token_to_id = {}
    id_to_token = {}
    for index, token in enumerate(tokens):
        token_to_id[token] = index
        id_to_token[index] = token

    transitions: Dict[Tuple[str, str], str] = {
        ("ROOT", OPEN_NODE): "ROUTE",
        ("ROUTE", TRANSITION_TOKEN): "SOURCE",
        ("ARGS_DONE", CLOSE_NODE): "TERMINAL",
        ("ARGS_DONE", MODIFIER_TOKEN): "MODIFIER",
        ("MODIFIER", CLOSE_NODE): "TERMINAL",
    }
    valid_tokens: Dict[str, Set[int]] = {
        "ROOT": {token_to_id[OPEN_NODE]},
        "ROUTE": {token_to_id[TRANSITION_TOKEN]},
        "SOURCE": {token_to_id[source] for source in sources},
        "ARGS_DONE": {token_to_id[CLOSE_NODE], token_to_id[MODIFIER_TOKEN]},
        "MODIFIER": {token_to_id[CLOSE_NODE]},
    }
    for source in sources:
        targets = targets_by_source[source]
        transitions[("SOURCE", source)] = "SOURCE_FROM_" + source
        valid_tokens["SOURCE_FROM_" + source] = {token_to_id[target] for target in targets}
        for target in targets:
            transitions[("SOURCE_FROM_" + source, target)] = "ARGS_DONE"

    grammar = SimpleASTGrammar(
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        node_types={MODIFIER_TOKEN: "OPTIONAL_MODIFIER"},
        transitions=transitions,
        valid_tokens=valid_tokens,
        closing_tokens={token_to_id[CLOSE_NODE]},
        terminal_nodes={"TERMINAL", "END"},
    )
    return grammar, token_to_id, id_to_token


def route_topology_text(text: str, binding: TopologyBinding) -> RoutedTree:
    """Route a sentence that names the endpoints of one declared directed edge."""
    mentions = mentioned_nodes(text, binding)
    if len(mentions) < 2:
        raise TextRoutingError(
            "the sentence names " + str(len(mentions)) + " declared node(s); naming a source and a "
            "destination is required. Declared nodes: " + ", ".join(binding.node_ids)
        )
    folded = fold_text(text)
    mentioned_ids = {node_id for node_id, _ in mentions}
    positions = {node_id: position for node_id, position in mentions}

    sources: List[str] = []
    targets_by_source: Dict[str, List[str]] = {}
    for source in binding.node_ids:
        if source not in mentioned_ids:
            continue
        targets = [
            target
            for target in binding.adjacency.get(source, [])
            if target in mentioned_ids and target != source
        ]
        if targets:
            sources.append(source)
            targets_by_source[source] = targets
    if not sources:
        raise TextRoutingError(
            "no declared route of the topology matches the named nodes. Declared routes: "
            + ", ".join(binding.routes())
        )

    grammar, token_to_id, id_to_token = topology_grammar(binding, sources, targets_by_source)
    mentioned: Dict[int, int] = {token_to_id[source]: positions[source] for source in sources}
    for source in sources:
        for target in targets_by_source[source]:
            mentioned[token_to_id[target]] = positions[target]
    modifier_position = first_mention(MODIFIER_WORDS, folded)
    if modifier_position >= 0:
        mentioned[token_to_id[MODIFIER_TOKEN]] = modifier_position

    scores = evidence_scores(token_to_id, folded, mentioned, vocab_size=len(token_to_id))
    runner = LexicalRunnerModel(
        vocab_size=len(token_to_id),
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        scores=scores,
    )
    tree, steps = run_engine_session(grammar, runner, prompt=text)
    validate_parentheses(tree)

    words = tree.split()
    if TRANSITION_TOKEN not in words or len(words) < 4:
        raise TextRoutingError("the decoded routing tree is incomplete: " + tree)
    source = words[words.index(TRANSITION_TOKEN) + 1]
    target = words[words.index(TRANSITION_TOKEN) + 2]
    if source not in binding.node_ids or target not in binding.node_ids:
        raise TextRoutingError("the decoded tree names a node that the topology does not declare")
    if not binding.route_exists(source, target):
        raise TextRoutingError("the decoded tree uses the undeclared route " + source + " -> " + target)
    for node_id in (source, target):
        if node_id not in mentioned_ids:
            raise TextRoutingError("the decoded tree names " + node_id + ", which the sentence never mentions")

    return RoutedTree(
        grammar=GRAMMAR_TOPOLOGY,
        tree=tree,
        skeleton=skeleton_of_parentheses(tree),
        symbols=words,
        steps=steps,
        evidence={
            "text": text,
            "sessionId": binding.session_id,
            "topologySource": binding.source_path,
            "mentionedNodes": [{"node": node_id, "position": positions[node_id]} for node_id, _ in mentions],
            "candidateRoutes": [
                source + " -> " + target
                for source in sorted(sources)
                for target in sorted(targets_by_source[source])
            ],
            "modifierMentioned": modifier_position >= 0,
            "route": {"source": source, "target": target, "declared": True},
        },
        valid=True,
    )


# ---------------------------------------------------------------------------
# Dataset route: trained weights plus the bracket algebra automaton
# ---------------------------------------------------------------------------
def resolve_model_package(path: Path) -> Path:
    """Find the self-contained release package inside a dataset or package folder."""
    root = Path(path)
    if not root.is_dir():
        raise TextRoutingError("model package folder not found: " + str(root))
    for candidate in PACKAGE_SEARCH_CANDIDATES:
        folder = (root / candidate).resolve()
        if not (folder / PACKAGE_TOKENIZER_FILE).is_file():
            continue
        if not (folder / PACKAGE_MASK_FILE).is_file():
            continue
        if any((folder / name).is_file() for name in PACKAGE_WEIGHT_CANDIDATES):
            return folder
    raise TextRoutingError(
        "no release package under " + str(root) + ": " + PACKAGE_TOKENIZER_FILE + ", "
        + PACKAGE_MASK_FILE + " and one of " + ", ".join(PACKAGE_WEIGHT_CANDIDATES) + " are required"
    )


class DatasetGrammarRouter:
    """Decode a sentence into a bracket tree with the weights and mask of a package."""

    def __init__(self, package: Path, device: Optional[str] = None):
        from neuro_symbolic.masking import AlgebraMask, TokenStreamMask
        from neuro_symbolic.tokenizer import Tokenizer

        folder = Path(package)
        self.package = folder
        self.tokenizer = Tokenizer.load(str(folder / PACKAGE_TOKENIZER_FILE))
        spec = json.loads((folder / PACKAGE_MASK_FILE).read_text(encoding="utf-8"))
        self.mask_spec = spec
        self.token_mask = TokenStreamMask(AlgebraMask.from_spec(spec["algebra"]), self.tokenizer)
        self.weights: Optional[Any] = None
        self.weights_name: str = ""
        self.torch = None
        self.device = device or "cpu"
        self._load_weights(folder)

    def _load_weights(self, folder: Path) -> None:
        safetensors = folder / "model.safetensors"
        torch_file = folder / "model.pt"
        try:
            import torch
        except ImportError as error:  # pragma: no cover - depends on the host
            raise TextRoutingError("the dataset grammar needs the torch package") from error
        self.torch = torch
        if safetensors.is_file():
            from neuro_symbolic.model import load_model_safetensors

            self.weights = load_model_safetensors(str(safetensors), device=self.device)
            self.weights_name = safetensors.name
        elif torch_file.is_file():
            from neuro_symbolic.model import load_model

            self.weights = load_model(str(torch_file), device=self.device)
            self.weights_name = torch_file.name
        else:
            raise TextRoutingError("no weights found in the package: " + str(folder))

    def describe(self) -> Dict[str, Any]:
        """Package identity, so a query says which weights and mask answered it."""
        return {
            "grammar": GRAMMAR_DATASET,
            "package": str(self.package),
            "weights": self.weights_name,
            "device": self.device,
            "vocabularySize": int(self.tokenizer.vocabulary_size),
            "operators": len(self.mask_spec.get("arities", {})),
            "packageFormat": self.mask_spec.get("package_format"),
        }

    def route(self, text: str, max_steps: Optional[int] = None) -> RoutedTree:
        """Greedy masked decoding of one sentence into the bracket algebra."""
        if not text.strip():
            raise TextRoutingError("--data is empty")
        torch = self.torch
        tokenizer = self.tokenizer
        source = torch.tensor([tokenizer.encode(text, add_bos=True)])
        attention = torch.ones_like(source, dtype=torch.bool)
        processor = NativeASTLogitsProcessor(vocab_size=int(tokenizer.vocabulary_size))
        rule_start = tokenizer.token_id("<RULE_START>")
        rule_end = tokenizer.token_id("<RULE_END>")
        end_token = tokenizer.token_id("<EOS>")
        limit = int(max_steps or self.weights.config.max_decoder_positions)
        prefix: List[int] = [rule_start]
        try:
            with torch.no_grad():
                memory = self.weights.encode(source, attention)
                while len(prefix) < limit:
                    logits = self.weights.decode(torch.tensor([prefix]), memory, attention)[0, -1]
                    processor.update_mask_from_ast_state(self.token_mask.allowed_next_ids(prefix))
                    masked = processor(prefix, logits.detach().cpu().numpy().astype(np.float32))
                    following = int(np.argmax(masked))
                    prefix.append(following)
                    if following in (rule_end, end_token):
                        break
                kernel = processor.is_native
        finally:
            processor.cleanup()

        symbols = tokenizer.decode_symbol_stream(prefix)
        try:
            tree = self.token_mask.mask.validate_stream(symbols)
        except Exception as error:  # AlgebraMaskError and tokenizer errors both refuse
            raise TextRoutingError("the decoded stream is not a valid tree: " + str(error)) from error
        word = " ".join(symbols)
        return RoutedTree(
            grammar=GRAMMAR_DATASET,
            tree=word,
            skeleton=self._skeleton(symbols),
            symbols=list(symbols),
            steps=len(prefix),
            evidence={
                "text": text,
                "encoderTokens": int(source.shape[1]),
                "treeJson": tree,
                "nativeKernel": bool(kernel),
            },
            model=self.describe(),
            valid=True,
        )

    @staticmethod
    def _skeleton(symbols: Sequence[str]) -> str:
        from neuro_symbolic.masking import skeleton_of_stream

        return skeleton_of_stream(symbols)


def route_dataset_text(text: str, package: Path, device: Optional[str] = None, max_steps: Optional[int] = None) -> RoutedTree:
    """Route a sentence through the model and mask of a release package."""
    return DatasetGrammarRouter(resolve_model_package(package), device=device).route(text, max_steps=max_steps)


# ---------------------------------------------------------------------------
# Routing entry point
# ---------------------------------------------------------------------------
def package_in_node_dir(node_dir: Path) -> Optional[Path]:
    """Return the release package a node directory carries, when it carries one."""
    node_dir = Path(node_dir)
    if not node_dir.is_dir():
        return None
    try:
        return resolve_model_package(node_dir)
    except TextRoutingError:
        return None


def route_text(
    text: str,
    grammar: str = GRAMMAR_AUTO,
    topology_config: Optional[Path] = None,
    model_package: Optional[Path] = None,
    dataset_root: Optional[Path] = None,
    node_dir: Optional[Path] = None,
    device: Optional[str] = None,
    max_steps: Optional[int] = None,
) -> RoutedTree:
    """Pick a grammar source and turn one sentence into a validated tree.

    With ``auto`` the order is: an explicitly named package, then the package of
    the node directory, then a declared topology, then the built-in alphabet. The
    node stays generic when nothing more specific is asked for.
    """
    if grammar not in GRAMMAR_CHOICES:
        raise TextRoutingError("unknown grammar source: " + str(grammar))
    if not text.strip():
        raise TextRoutingError("--data is empty")

    selected = grammar
    if grammar == GRAMMAR_AUTO:
        if model_package is not None or dataset_root is not None:
            selected = GRAMMAR_DATASET
        elif node_dir is not None and package_in_node_dir(node_dir) is not None:
            selected = GRAMMAR_DATASET
        elif topology_config is not None:
            selected = GRAMMAR_TOPOLOGY
        else:
            selected = GRAMMAR_GENERIC

    if selected == GRAMMAR_DATASET:
        origin = model_package or dataset_root or node_dir
        if origin is None:
            raise TextRoutingError("the dataset grammar needs --model-package or --dataset-root")
        return route_dataset_text(text, Path(origin), device=device, max_steps=max_steps)

    if selected == GRAMMAR_TOPOLOGY:
        if topology_config is None:
            raise TextRoutingError("the topology grammar needs --topology-config PATH")
        return route_topology_text(text, load_topology_binding(Path(topology_config)))

    return route_generic_text(text)


def render_report(result: RoutedTree) -> str:
    """Human readable rendering of one routing result for the console."""
    lines = [
        "grammar: " + result.grammar,
        "tree:    " + result.tree,
        "skeleton: " + result.skeleton,
        "valid:   " + ("yes" if result.valid else "no") + "  steps: " + str(result.steps),
    ]
    route = result.evidence.get("route")
    if route:
        lines.append("route:   " + str(route.get("source")) + " -> " + str(route.get("target")))
    if result.model:
        lines.append(
            "model:   " + str(result.model.get("package")) + " " + str(result.model.get("weights"))
            + " vocab=" + str(result.model.get("vocabularySize"))
            + " operators=" + str(result.model.get("operators"))
        )
    mentions = result.evidence.get("mentionedNodes")
    if mentions:
        lines.append("nodes:   " + ", ".join(str(item["node"]) for item in mentions))
    return "\n".join(lines)
