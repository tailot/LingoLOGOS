"""Independent reference model for the asic_chip checks.

This module is not collected by pytest because its name does not start with test_.
It carries the second implementation of every rule the asic_chip tests depend on,
written without reusing asic_chip/scripts/lingologos_compiler.py, so that agreement
between the two is evidence and not a tautology:

  * GF(2^8) field arithmetic modulo the primitive polynomial 0x11D, done with the
    shift and conditional xor method instead of the table method,
  * irreducibility of a GF(2) polynomial through the gcd criterion
    gcd(f, x^(2^k) - x) = 1 for 1 <= k <= deg(f)/2 together with x^(2^n) = x,
  * the weight mesh as a plain signed dot product, with no non-adjacent form and no
    shift add decomposition anywhere,
  * the clause DSL of asic_chip/artifacts/smt_constraints.json parsed with the
    shunting-yard algorithm into reverse polish form and evaluated as full 256 entry
    truth tables, a different parsing strategy and a different evaluation strategy
    from the recursive descent netlist builder in the compiler,
  * a cycle accurate interpreter for the RV32E firmware stored in the program memory
    of asic_chip/rtl/core/rv32e_core.sv, including the coprocessor contract of
    asic_chip/rtl/top/lingologos_top.sv.

The acceptance gates live in the test files that import this model.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

# -----------------------------------------------------------------------------
# GF(2^8) field arithmetic
# -----------------------------------------------------------------------------

DEFAULT_FIELD_POLYNOMIAL = 0x11D


def gf_mul(a: int, b: int, polynomial: int = DEFAULT_FIELD_POLYNOMIAL) -> int:
    """Multiply two bytes in GF(2^8), Russian peasant with conditional reduction."""
    left = a & 0xFF
    right = b & 0xFF
    result = 0
    for _ in range(8):
        if right & 1:
            result ^= left
        right >>= 1
        carry_out = left & 0x80
        left = (left << 1) & 0xFF
        if carry_out:
            left ^= polynomial & 0xFF
    return result & 0xFF


def gf_pow(base: int, exponent: int, polynomial: int = DEFAULT_FIELD_POLYNOMIAL) -> int:
    result = 1
    value = base & 0xFF
    remaining = exponent
    while remaining > 0:
        if remaining & 1:
            result = gf_mul(result, value, polynomial)
        value = gf_mul(value, value, polynomial)
        remaining >>= 1
    return result


def gf_order(base: int, polynomial: int = DEFAULT_FIELD_POLYNOMIAL) -> int:
    """Multiplicative order of a nonzero field element, found by iterating."""
    if base == 0:
        raise ValueError("zero has no multiplicative order")
    value = base & 0xFF
    order = 1
    while value != 1:
        value = gf_mul(value, base, polynomial)
        order += 1
        if order > 1000:
            raise ValueError("element does not return to one inside the field")
    return order


def gf_inverse(base: int, polynomial: int = DEFAULT_FIELD_POLYNOMIAL) -> int:
    """Inverse by brute force over the 255 nonzero elements."""
    if base == 0:
        raise ValueError("zero has no inverse")
    for candidate in range(1, 256):
        if gf_mul(base, candidate, polynomial) == 1:
            return candidate
    raise ValueError("no inverse found, the modulus is not a field")


# -----------------------------------------------------------------------------
# GF(2) polynomial arithmetic, used for the irreducibility check
# -----------------------------------------------------------------------------


def gf2_degree(poly: int) -> int:
    degree = -1
    value = poly
    while value:
        degree += 1
        value >>= 1
    return degree


def gf2_poly_mul(a: int, b: int) -> int:
    result = 0
    for bit in range(gf2_degree(b) + 1):
        if (b >> bit) & 1:
            result ^= a << bit
    return result


def gf2_poly_mod(dividend: int, modulus: int) -> int:
    remainder = dividend
    mod_degree = gf2_degree(modulus)
    while gf2_degree(remainder) >= mod_degree and remainder:
        shift = gf2_degree(remainder) - mod_degree
        remainder ^= modulus << shift
    return remainder


def gf2_gcd(a: int, b: int) -> int:
    left, right = a, b
    while right:
        left, right = right, gf2_poly_mod(left, right)
    return left


def gf2_x_pow_2k_mod(k: int, modulus: int) -> int:
    """x raised to 2^k, reduced modulo the given polynomial."""
    value = 0b10 % modulus if gf2_degree(0b10) >= gf2_degree(modulus) else 0b10
    for _ in range(k):
        value = gf2_poly_mod(gf2_poly_mul(value, value), modulus)
    return value


def gf2_is_irreducible(modulus: int) -> bool:
    """Full gcd criterion: x^(2^n) = x mod f and no factor of degree at most n/2."""
    degree = gf2_degree(modulus)
    if degree < 1:
        return False
    if gf2_x_pow_2k_mod(degree, modulus) != 0b10:
        return False
    for k in range(1, degree // 2 + 1):
        difference = gf2_x_pow_2k_mod(k, modulus) ^ 0b10
        if gf2_gcd(modulus, difference) != 1:
            return False
    return True


# -----------------------------------------------------------------------------
# Signed helpers and the weight mesh
# -----------------------------------------------------------------------------


def to_signed(value: int, bits: int) -> int:
    masked = value & ((1 << bits) - 1)
    return masked - (1 << bits) if masked & (1 << (bits - 1)) else masked


def to_unsigned(value: int, bits: int) -> int:
    return value & ((1 << bits) - 1)


def mesh_row_sum(values: Sequence[Sequence[int]], row: int, inputs: Sequence[int]) -> int:
    """Row of the mesh as a plain signed dot product of the reference weights."""
    total = 0
    for weight, sample in zip(values[row], inputs):
        total += int(weight) * int(sample)
    return total


# -----------------------------------------------------------------------------
# The clause DSL, parsed with shunting-yard and evaluated as full truth tables
# -----------------------------------------------------------------------------

NOT_PRECEDENCE = 4
INFIX_PRECEDENCE = {"MUX": 0, "OR": 1, "XOR": 2, "AND": 3}
NAMED_FUNCTIONS = ("AND", "OR", "XOR", "MUX")
CALL_MAX_ARITY = {"AND": 32, "OR": 32, "XOR": 32, "MUX": 3}
CALL_MIN_ARITY = {"AND": 2, "OR": 2, "XOR": 2, "MUX": 3}


def tokenize_formula(text: str) -> List[str]:
    tokens: List[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
        elif char in "()":
            tokens.append(char)
            index += 1
        elif char == ",":
            tokens.append(",")
            index += 1
        elif char.isalpha():
            stop = index
            while stop < len(text) and text[stop].isalpha():
                stop += 1
            tokens.append(text[index:stop].upper())
            index = stop
        elif char.isdigit():
            tokens.append(char)
            index += 1
        else:
            raise ValueError(f"unexpected character in clause: {char!r}")
    return tokens


def formula_to_rpn(tokens: Sequence[str]) -> List[Tuple[str, object]]:
    """Shunting-yard conversion; a call frame also delimits its own parentheses.

    The same word can act as an infix operator or as the head of a call, depending on
    whether an operand or an operator is expected at that point, which is how the
    documented DSL reads: A AND B is infix, AND(A, B) is a call, NOT C is prefix and
    value MUX select is the lowest precedence infix form.
    """
    output: List[Tuple[str, object]] = []
    # Stack frames: ["lparen"], ["func", name, separators], ["op", name, precedence].
    frames: List[List[object]] = []
    expecting_operand = True
    call_needs_paren = False

    def emit_top_operator() -> None:
        frame = frames.pop()
        name = str(frame[1])
        output.append((name, 1 if name == "NOT" else 2))

    def close_group() -> None:
        while frames and frames[-1][0] == "op":
            emit_top_operator()
        if not frames:
            raise ValueError("unbalanced parenthesis in clause")
        frame = frames.pop()
        if frame[0] == "lparen":
            return
        name = str(frame[1])
        arity = int(frame[2]) + 1
        if arity < CALL_MIN_ARITY[name] or arity > CALL_MAX_ARITY[name]:
            raise ValueError(f"{name} takes {CALL_MIN_ARITY[name]} to {CALL_MAX_ARITY[name]} operands, got {arity}")
        output.append((name, arity))

    for token in tokens:
        if call_needs_paren and token != "(":
            raise ValueError("a function name must be followed by a left parenthesis")
        if token == "(":
            if call_needs_paren:
                call_needs_paren = False
            else:
                if not expecting_operand:
                    raise ValueError("unexpected left parenthesis in clause")
                frames.append(["lparen"])
            expecting_operand = True
        elif token in NAMED_FUNCTIONS:
            if expecting_operand:
                frames.append(["func", token, 0])
                call_needs_paren = True
            else:
                precedence = INFIX_PRECEDENCE[token]
                while frames and frames[-1][0] == "op" and int(frames[-1][2]) >= precedence:
                    emit_top_operator()
                frames.append(["op", token, precedence])
            expecting_operand = True
        elif token == "NOT":
            if not expecting_operand:
                raise ValueError("NOT used where an operator was expected")
            frames.append(["op", "NOT", NOT_PRECEDENCE])
            expecting_operand = True
        elif token == ",":
            while frames and frames[-1][0] == "op":
                emit_top_operator()
            if not frames or frames[-1][0] != "func":
                raise ValueError("comma outside a function call")
            frames[-1][2] = int(frames[-1][2]) + 1
            expecting_operand = True
        elif token == ")":
            if expecting_operand:
                raise ValueError("empty group in clause")
            close_group()
            expecting_operand = False
        else:
            if not expecting_operand:
                raise ValueError("two operands in a row in clause")
            output.append(("leaf", token))
            expecting_operand = False
    while frames:
        if frames[-1][0] == "lparen":
            raise ValueError("unbalanced parenthesis in clause")
        if frames[-1][0] == "func":
            raise ValueError("a function call is missing its closing parenthesis")
        emit_top_operator()
    return output


def variable_pattern(bit: int, input_bits: int) -> int:
    """256 bit mask of the assignments in which the given input bit is one."""
    mask = 0
    for assignment in range(1 << input_bits):
        if (assignment >> bit) & 1:
            mask |= 1 << assignment
    return mask


def evaluate_rpn(rpn: Sequence[Tuple[str, object]], variable_bits: Dict[str, int], input_bits: int) -> int:
    """Evaluate the reverse polish form as a mask over all 2^input_bits assignments."""
    all_assignments = (1 << (1 << input_bits)) - 1
    patterns = {name: variable_pattern(bit, input_bits) for name, bit in variable_bits.items()}
    stack: List[int] = []
    for entry, argument in rpn:
        if entry == "leaf":
            name = str(argument)
            if name in ("0", "1"):
                stack.append(0 if name == "0" else all_assignments)
            elif name in patterns:
                stack.append(patterns[name])
            else:
                raise ValueError(f"undeclared variable in clause: {name}")
            continue
        arity = int(argument)
        if entry == "MUX" and arity == 3:
            clear_value, set_value, selector = stack.pop(), stack.pop(), stack.pop()
            # select ? set : clear, as a mask identity: flip clear wherever select is set.
            stack.append(((set_value ^ clear_value) & selector) ^ clear_value)
            continue
        if entry == "MUX" and arity == 2:
            selector, value = stack.pop(), stack.pop()
            stack.append(value & selector)
            continue
        if entry == "MUX":
            raise ValueError("prefix MUX takes exactly three operands")
        if entry == "NOT":
            if arity != 1:
                raise ValueError("NOT takes exactly one operand")
            stack.append(~stack.pop() & all_assignments)
            continue
        if len(stack) < arity:
            raise ValueError(f"{entry} is missing operands")
        accumulator = stack.pop()
        for _ in range(arity - 1):
            operand = stack.pop()
            if entry == "AND":
                accumulator &= operand
            elif entry == "OR":
                accumulator |= operand
            elif entry == "XOR":
                accumulator ^= operand
            else:
                raise ValueError(f"unknown operator {entry}")
        stack.append(accumulator)
    if len(stack) != 1:
        raise ValueError("clause does not reduce to a single value")
    return stack[0]


def clause_mask(expression: str, variable_bits: Dict[str, int], input_bits: int) -> int:
    return evaluate_rpn(formula_to_rpn(tokenize_formula(expression)), variable_bits, input_bits)


def clause_truth_table(expression: str, variable_bits: Dict[str, int], input_bits: int) -> List[int]:
    mask = clause_mask(expression, variable_bits, input_bits)
    return [(mask >> assignment) & 1 for assignment in range(1 << input_bits)]


# -----------------------------------------------------------------------------
# RV32E interpreter for the firmware in the core program memory
# -----------------------------------------------------------------------------

OPC_LUI = 0b0110111
OPC_AUIPC = 0b0010111
OPC_JAL = 0b1101111
OPC_JALR = 0b1100111
OPC_BRANCH = 0b1100011
OPC_LOAD = 0b0000011
OPC_STORE = 0b0100011
OPC_MISC_MEM = 0b0001111
OPC_OP_IMM = 0b0010011
OPC_OP = 0b0110011
OPC_CUSTOM = 0b0001011

COP_SMT = 0
COP_GF = 1
COP_MESH = 2
COP_STAT = 3

# Which opcodes read which register field. Bits 24:20 of an OP-IMM are immediate bits
# and bits 11:7 of a STORE or BRANCH are immediate bits, so those slots must not be
# checked as register names even though they share a position with one.
RD_OPCODES = frozenset({OPC_LUI, OPC_AUIPC, OPC_JAL, OPC_JALR, OPC_LOAD, OPC_OP_IMM, OPC_OP, OPC_CUSTOM})
RS1_OPCODES = frozenset({OPC_BRANCH, OPC_LOAD, OPC_STORE, OPC_OP_IMM, OPC_OP, OPC_JALR, OPC_CUSTOM})
RS2_OPCODES = frozenset({OPC_BRANCH, OPC_STORE, OPC_OP})


def encoding_is_legal(opcode: int, funct3: int, funct7: int) -> bool:
    """Mirror of the documented decoder rules, written independently."""
    if opcode in (OPC_LUI, OPC_AUIPC, OPC_JAL, OPC_JALR):
        return True
    if opcode == OPC_BRANCH:
        return funct3 not in (0b010, 0b011)
    if opcode == OPC_LOAD:
        return funct3 not in (0b011, 0b110, 0b111)
    if opcode == OPC_STORE:
        return funct3 in (0b000, 0b001, 0b010)
    if opcode == OPC_OP_IMM:
        if funct3 == 0b001:
            return funct7 == 0b0000000
        if funct3 == 0b101:
            return funct7 in (0b0000000, 0b0100000)
        return True
    if opcode == OPC_OP:
        if funct3 in (0b000, 0b101):
            return funct7 in (0b0000000, 0b0100000)
        return funct7 == 0b0000000
    if opcode == OPC_MISC_MEM:
        return funct3 == 0b000
    if opcode == OPC_CUSTOM:
        return funct3 in (0b001, 0b010, 0b011, 0b100, 0b111)
    return False


def register_space_is_legal(opcode: int, rd: int, rs1: int, rs2: int) -> bool:
    """RV32E register space rule, applied only to the fields the opcode reads."""
    if opcode in RD_OPCODES and rd > 15:
        return False
    if opcode in RS1_OPCODES and rs1 > 15:
        return False
    if opcode in RS2_OPCODES and rs2 > 15:
        return False
    return True


def access_is_aligned(funct3: int, address: int) -> bool:
    if funct3 in (0b001, 0b101):
        return address & 1 == 0
    if funct3 == 0b010:
        return address & 3 == 0
    return True


class CoprocessorModel:
    """Contract of the coprocessor as wired in the top level of the design."""

    def __init__(
        self,
        weight_values: Sequence[Sequence[int]],
        gf_initial_state: int,
        gf_multipliers: Dict[int, int],
        clause_tables: Sequence[Sequence[int]],
        pad_inputs: Sequence[int],
        pad_symbol: int,
    ) -> None:
        self.weight_values = weight_values
        self.gf_initial_state = gf_initial_state
        self.gf_multipliers = gf_multipliers
        self.clause_tables = clause_tables
        self.pad_inputs = pad_inputs
        self.pad_symbol = pad_symbol
        self.gf_state = gf_initial_state
        self.cop_count = 0

    def advance_automaton(self, symbol: int) -> None:
        multiplier = self.gf_multipliers.get(symbol & 0xFF)
        if multiplier is None:
            self.gf_state = self.gf_initial_state
        else:
            self.gf_state = gf_mul(self.gf_state, multiplier)

    def gf_next(self, symbol: int) -> int:
        multiplier = self.gf_multipliers.get(symbol & 0xFF)
        if multiplier is None:
            return self.gf_initial_state
        return gf_mul(self.gf_state, multiplier)

    def read(self, sel: int, rs1_value: int, illegal_seen: int) -> Tuple[int, bool]:
        """Return the 32 bit readback and whether the coprocessor counter moves."""
        if sel == COP_SMT:
            byte = rs1_value & 0xFF
            word = 0
            for lane, table in enumerate(self.clause_tables):
                word |= (table[byte] & 1) << lane
            return word, True
        if sel == COP_GF:
            return self.gf_next(rs1_value & 0xFF), True
        if sel == COP_MESH:
            row = rs1_value & 3
            return to_unsigned(mesh_row_sum(self.weight_values, row, self.pad_inputs), 32), True
        if sel == COP_STAT:
            return ((illegal_seen & 1) << 4) | (self.cop_count & 0xF), True
        raise ValueError("unknown coprocessor select")


class CoreRun:
    """Result of one emulated self-test run."""

    def __init__(self) -> None:
        self.registers: List[int] = [0] * 16
        self.illegal_words: List[int] = []
        self.executed_words: List[int] = []
        self.cycles = 0
        self.halted = False
        self.pass_flag = 0
        self.cop_count = 0
        self.gf_final_state = 0
        self.dmem: List[int] = []


def run_firmware(
    rom: Dict[int, int],
    weight_values: Sequence[Sequence[int]],
    gf_initial_state: int,
    gf_multipliers: Dict[int, int],
    clause_tables: Sequence[Sequence[int]],
    pad_inputs: Sequence[int],
    pad_symbol: int,
    imem_words: int,
    dmem_words: int,
    cycle_limit: int = 800,
) -> CoreRun:
    """Execute the program memory the way the single-cycle core is documented to."""
    run = CoreRun()
    registers = run.registers
    data_ram = [0] * dmem_words
    cop = CoprocessorModel(
        weight_values, gf_initial_state, gf_multipliers, clause_tables, pad_inputs, pad_symbol
    )
    pc = 0
    illegal_seen = 0
    while not run.halted and run.cycles < cycle_limit:
        run.cycles += 1
        word_index = (pc >> 2) % imem_words
        instruction = rom[word_index]
        run.executed_words.append(word_index)

        opcode = instruction & 0x7F
        funct3 = (instruction >> 12) & 0b111
        funct7 = (instruction >> 25) & 0x7F
        rd = (instruction >> 7) & 0x1F
        rs1_index = (instruction >> 15) & 0x1F
        rs2_index = (instruction >> 20) & 0x1F
        shamt = (instruction >> 20) & 0x1F

        def sign_extend(value: int, bits: int) -> int:
            return to_signed(value, bits)

        imm_i = sign_extend(instruction >> 20, 12)
        imm_s = sign_extend(((instruction >> 25) << 5) | ((instruction >> 7) & 0x1F), 12)
        imm_b = sign_extend(
            ((instruction >> 31) << 11)
            | (((instruction >> 7) & 1) << 10)
            | (((instruction >> 25) & 0x3F) << 4)
            | (((instruction >> 8) & 0xF) << 1),
            12,
        )
        imm_u = (instruction & 0xFFFFF000)
        imm_j = sign_extend(
            ((instruction >> 31) << 19)
            | (((instruction >> 12) & 0xFF) << 11)
            | (((instruction >> 20) & 1) << 10)
            | (((instruction >> 21) & 0x3FF) << 1),
            20,
        )

        register_space_ok = register_space_is_legal(opcode, rd, rs1_index, rs2_index)
        legal = encoding_is_legal(opcode, funct3, funct7) and register_space_ok
        memory_access = opcode in (OPC_LOAD, OPC_STORE)
        rs1_value = registers[rs1_index & 0xF] if rs1_index < 16 else 0
        rs2_value = registers[rs2_index & 0xF] if rs2_index < 16 else 0
        load_address = to_unsigned(rs1_value + imm_i, 32)
        store_address = to_unsigned(rs1_value + imm_s, 32)
        access_address = store_address if opcode == OPC_STORE else load_address
        if legal and memory_access:
            legal = access_is_aligned(funct3, access_address)

        if not legal:
            illegal_seen = 1
            run.illegal_words.append(word_index)
            cop.advance_automaton(pad_symbol)
            pc = to_unsigned(pc + 4, 32) % (4 * imem_words)
            continue

        next_pc = to_unsigned(pc + 4, 32) % (4 * imem_words)
        write_value = None
        cop_word = None
        cop_moves = False
        gf_step_symbol = None
        halting = False

        if opcode == OPC_LUI:
            write_value = imm_u
        elif opcode == OPC_AUIPC:
            write_value = to_unsigned(pc + imm_u, 32)
        elif opcode == OPC_JAL:
            write_value = to_unsigned(pc + 4, 32)
            next_pc = to_unsigned(pc + imm_j, 32) % (4 * imem_words)
        elif opcode == OPC_JALR:
            write_value = to_unsigned(pc + 4, 32)
            next_pc = to_unsigned((rs1_value + imm_i) & ~1, 32) % (4 * imem_words)
        elif opcode == OPC_BRANCH:
            taken = (
                (funct3 == 0b000 and rs1_value == rs2_value)
                or (funct3 == 0b001 and rs1_value != rs2_value)
                or (funct3 == 0b100 and to_signed(rs1_value, 32) < to_signed(rs2_value, 32))
                or (funct3 == 0b101 and to_signed(rs1_value, 32) >= to_signed(rs2_value, 32))
                or (funct3 == 0b110 and rs1_value < rs2_value)
                or (funct3 == 0b111 and rs1_value >= rs2_value)
            )
            if taken:
                next_pc = to_unsigned(pc + imm_b, 32) % (4 * imem_words)
        elif opcode == OPC_LOAD:
            index = (load_address >> 2) % dmem_words
            read_word = data_ram[index]
            lane = load_address & 3
            half_lane = ((load_address >> 1) & 1) * 8
            if funct3 == 0b000:
                write_value = to_signed((read_word >> (lane * 8)) & 0xFF, 8)
            elif funct3 == 0b001:
                write_value = to_signed((read_word >> half_lane) & 0xFFFF, 16)
            elif funct3 == 0b010:
                write_value = read_word
            elif funct3 == 0b100:
                write_value = (read_word >> (lane * 8)) & 0xFF
            else:
                write_value = (read_word >> half_lane) & 0xFFFF
            write_value = to_unsigned(write_value, 32)
        elif opcode == OPC_STORE:
            index = (store_address >> 2) % dmem_words
            lane = store_address & 3
            half_lane = ((store_address >> 1) & 1) * 8
            if funct3 == 0b000:
                data_ram[index] = to_unsigned(
                    (data_ram[index] & ~(0xFF << (lane * 8))) | ((rs2_value & 0xFF) << (lane * 8)), 32
                )
            elif funct3 == 0b001:
                data_ram[index] = to_unsigned(
                    (data_ram[index] & ~(0xFFFF << half_lane)) | ((rs2_value & 0xFFFF) << half_lane), 32
                )
            else:
                data_ram[index] = to_unsigned(rs2_value, 32)
        elif opcode == OPC_OP_IMM or opcode == OPC_OP:
            operand_b = rs2_value if opcode == OPC_OP else imm_i
            if funct3 == 0b000:
                subtract = (opcode == OPC_OP) and bool((instruction >> 30) & 1)
                write_value = to_unsigned(rs1_value - operand_b if subtract else rs1_value + operand_b, 32)
            elif funct3 == 0b001:
                write_value = to_unsigned(rs1_value << shamt, 32)
            elif funct3 == 0b010:
                write_value = 1 if to_signed(rs1_value, 32) < to_signed(operand_b, 32) else 0
            elif funct3 == 0b011:
                write_value = 1 if rs1_value < to_unsigned(operand_b, 32) else 0
            elif funct3 == 0b100:
                write_value = to_unsigned(rs1_value ^ operand_b, 32)
            elif funct3 == 0b101:
                if (instruction >> 30) & 1:
                    write_value = to_unsigned(to_signed(rs1_value, 32) >> shamt, 32)
                else:
                    write_value = to_unsigned(rs1_value >> shamt, 32)
            elif funct3 == 0b110:
                write_value = to_unsigned(rs1_value | operand_b, 32)
            else:
                write_value = to_unsigned(rs1_value & operand_b, 32)
        elif opcode == OPC_MISC_MEM:
            pass
        elif opcode == OPC_CUSTOM:
            if funct3 == 0b111:
                halting = True
                run.pass_flag = rs1_value & 1
            else:
                sel = {0b001: COP_SMT, 0b010: COP_GF, 0b011: COP_MESH, 0b100: COP_STAT}[funct3]
                cop_word, cop_moves = cop.read(sel, rs1_value, illegal_seen)
                write_value = cop_word
                if sel == COP_GF:
                    gf_step_symbol = rs1_value & 0xFF
        else:
            illegal_seen = 1
            run.illegal_words.append(word_index)
            cop.advance_automaton(pad_symbol)
            pc = to_unsigned(pc + 4, 32) % (4 * imem_words)
            continue

        if write_value is not None and rd != 0:
            registers[rd] = to_unsigned(write_value, 32)
        if gf_step_symbol is not None:
            cop.gf_state = cop.gf_next(gf_step_symbol)
        else:
            cop.advance_automaton(pad_symbol)
        if cop_moves:
            cop.cop_count = (cop.cop_count + 1) & 0xF
        run.halted = halting
        if not halting:
            pc = next_pc

    run.illegal_seen = illegal_seen
    run.cop_count = cop.cop_count
    run.gf_final_state = cop.gf_state
    run.dmem = data_ram
    return run
