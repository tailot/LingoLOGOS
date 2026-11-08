// -----------------------------------------------------------------------------
// C++ harness for the LingoLOGOS ASIC, built against a Verilated model.
//
// Section 8 of the LingoLOGOS ASIC specification asks for a C++ harness under
// sim/sim_main.cpp that drives the chip from the same golden vectors both simulators
// consume. The vectors live in asic_chip/sim/golden_vectors.mem and
// asic_chip/sim/core_self_test.mem, both emitted by
// asic_chip/scripts/lingologos_compiler.py, so the reference numbers come from the
// artifacts and not from anything written by hand in this file.
//
// Usage, after Verilator has produced the model class:
//   make -f asic_chip/Makefile sim-verilator
// or by hand:
//   obj_dir/Vlingologos_top asic_chip/sim/golden_vectors.mem \
//                          asic_chip/sim/core_self_test.mem lingologos.vcd
//
// Timing protocol, identical to sim/testbench.sv:
//   inputs change while clk is low; smt_results is checked in that same cycle
//   because the clause network is combinational; one clock edge later out_vector_*
//   and gf_next_state are checked because both are registered.
//
// The second phase freezes the pads, starts the RV32E instruction-set self-test and
// checks the coprocessor readbacks the firmware deposits in its registers.
// -----------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "Vlingologos_top.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

namespace {

const int kColumns = 12;

struct Vector {
    int in[4];
    uint8_t gf_symbol;
    uint8_t smt_bits;
    long expected_out[4];
    uint8_t expected_gf_state;
    uint8_t expected_smt;
};

// Sign extend a value that was read as an unsigned column of the given bit width.
long sign_extend(long value, int bits) {
    const long limit = 1L << (bits - 1);
    if (value >= limit) {
        return value - (1L << bits);
    }
    return value;
}

int bit_width_of_hex(const std::string& text) {
    return static_cast<int>(text.size()) * 4;
}

bool is_comment_or_blank(const std::string& line) {
    size_t index = 0;
    while (index < line.size() && (line[index] == ' ' || line[index] == '\t')) {
        ++index;
    }
    return index >= line.size() || line.compare(index, 2, "//") == 0;
}

std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> parts;
    std::istringstream stream(line);
    std::string token;
    while (stream >> token) {
        parts.push_back(token);
    }
    return parts;
}

// A // tail is a comment even on a data line, so cut it before splitting.
std::string strip_comment(const std::string& line) {
    const size_t found = line.find("//");
    if (found == std::string::npos) {
        return line;
    }
    return line.substr(0, found);
}

long parse_hex(const std::string& text) {
    return std::strtol(text.c_str(), nullptr, 16);
}

bool load_vectors(const std::string& path, std::vector<Vector>* vectors, std::string* error) {
    std::ifstream file(path.c_str());
    if (!file.is_open()) {
        *error = "cannot open " + path;
        return false;
    }
    std::string line;
    int row = 0;
    while (std::getline(file, line)) {
        const std::string body = strip_comment(line);
        if (is_comment_or_blank(body)) {
            continue;
        }
        const std::vector<std::string> tokens = split(body);
        if (static_cast<int>(tokens.size()) != kColumns) {
            std::ostringstream message;
            message << "row " << row << " of " << path << " has " << tokens.size()
                    << " columns, expected " << kColumns;
            *error = message.str();
            return false;
        }
        Vector vector;
        vector.in[0] = static_cast<int>(sign_extend(parse_hex(tokens[0]), bit_width_of_hex(tokens[0])));
        vector.in[1] = static_cast<int>(sign_extend(parse_hex(tokens[1]), bit_width_of_hex(tokens[1])));
        vector.in[2] = static_cast<int>(sign_extend(parse_hex(tokens[2]), bit_width_of_hex(tokens[2])));
        vector.in[3] = static_cast<int>(sign_extend(parse_hex(tokens[3]), bit_width_of_hex(tokens[3])));
        vector.gf_symbol = static_cast<uint8_t>(parse_hex(tokens[4]));
        vector.smt_bits = static_cast<uint8_t>(parse_hex(tokens[5]));
        for (int index = 0; index < 4; ++index) {
            const std::string& column = tokens[6 + index];
            vector.expected_out[index] = sign_extend(parse_hex(column), bit_width_of_hex(column));
        }
        vector.expected_gf_state = static_cast<uint8_t>(parse_hex(tokens[10]));
        vector.expected_smt = static_cast<uint8_t>(parse_hex(tokens[11]));
        vectors->push_back(vector);
        ++row;
    }
    if (vectors->empty()) {
        *error = "no data rows in " + path;
        return false;
    }
    return true;
}

bool load_self_test(const std::string& path, std::map<std::string, long>* values, std::string* error) {
    std::ifstream file(path.c_str());
    if (!file.is_open()) {
        *error = "cannot open " + path;
        return false;
    }
    std::string line;
    while (std::getline(file, line)) {
        const std::string body = strip_comment(line);
        if (is_comment_or_blank(body)) {
            continue;
        }
        const std::vector<std::string> tokens = split(body);
        if (tokens.size() != 3 || tokens[1] != "=") {
            std::ostringstream message;
            message << "expected key = value in " << path << ", got: " << body;
            *error = message.str();
            return false;
        }
        (*values)[tokens[0]] = parse_hex(tokens[2]);
    }
    const char* required[] = {"in_vector_0", "in_vector_1", "in_vector_2", "in_vector_3",
                              "gf_symbol_in", "smt_in_bits", "exp_x8", "exp_x10", "exp_x11",
                              "exp_x13", "exp_x15", "exp_core_pass", "exp_core_illegal_seen",
                              "exp_cop_count", "exp_gf_state_final"};
    for (size_t index = 0; index < sizeof(required) / sizeof(required[0]); ++index) {
        if (values->find(required[index]) == values->end()) {
            *error = std::string("missing key ") + required[index] + " in " + path;
            return false;
        }
    }
    return true;
}

long g_failure_count = 0;

void check(bool condition, const std::string& what, long got, long want) {
    if (condition) {
        return;
    }
    ++g_failure_count;
    std::cout << "[FAIL] " << what << ": got 0x" << std::hex << got << " want 0x" << want
              << std::dec << std::endl;
}

void check_signed(long got, long want, const std::string& what) {
    check(got == want, what, got, want);
}

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    std::string golden_path = "asic_chip/sim/golden_vectors.mem";
    std::string selftest_path = "asic_chip/sim/core_self_test.mem";
    std::string vcd_path = "";
    if (argc > 1) {
        golden_path = argv[1];
    }
    if (argc > 2) {
        selftest_path = argv[2];
    }
    if (argc > 3) {
        vcd_path = argv[3];
    }

    std::vector<Vector> vectors;
    std::map<std::string, long> selftest;
    std::string error;
    if (!load_vectors(golden_path, &vectors, &error) || !load_self_test(selftest_path, &selftest, &error)) {
        std::cout << "[FAIL] " << error << std::endl;
        std::cout << "FINAL VERDICT: FAIL" << std::endl;
        return 1;
    }

    Vlingologos_top* chip = new Vlingologos_top;
    VerilatedVcdC* trace = nullptr;
    if (!vcd_path.empty()) {
        Verilated::traceEverOn(true);
        trace = new VerilatedVcdC;
        chip->trace(trace, 99);
        trace->open(vcd_path.c_str());
    }

    // Reset: two cycles with rst_n low, then release.
    uint64_t tick = 0;
    chip->rst_n = 0;
    chip->core_start = 0;
    chip->in_vector_0 = 0;
    chip->in_vector_1 = 0;
    chip->in_vector_2 = 0;
    chip->in_vector_3 = 0;
    chip->gf_symbol_in = 0;
    chip->smt_in_bits = 0;
    chip->clk = 0;
    for (int cycle = 0; cycle < 4; ++cycle) {
        chip->clk = 0;
        chip->eval();
        if (trace) { trace->dump(static_cast<vluint64_t>(tick++)); }
        chip->clk = 1;
        chip->eval();
        if (trace) { trace->dump(static_cast<vluint64_t>(tick++)); }
    }
    chip->rst_n = 1;

    // Phase A: the golden vector table.
    std::cout << "[INFO] phase A, " << vectors.size() << " golden rows from " << golden_path
              << std::endl;
    for (size_t row = 0; row < vectors.size(); ++row) {
        const Vector& vector = vectors[row];
        chip->clk = 0;
        chip->eval();
        chip->in_vector_0 = static_cast<uint8_t>(vector.in[0] & 0xFF);
        chip->in_vector_1 = static_cast<uint8_t>(vector.in[1] & 0xFF);
        chip->in_vector_2 = static_cast<uint8_t>(vector.in[2] & 0xFF);
        chip->in_vector_3 = static_cast<uint8_t>(vector.in[3] & 0xFF);
        chip->gf_symbol_in = vector.gf_symbol;
        chip->smt_in_bits = vector.smt_bits;
        chip->eval();

        check_signed(chip->smt_results, vector.expected_smt,
                     "row " + std::to_string(row) + " smt_results (combinational)");

        chip->clk = 1;
        chip->eval();
        if (trace) { trace->dump(static_cast<vluint64_t>(tick++)); }

        check_signed(sign_extend(chip->out_vector_0, 20), vector.expected_out[0],
                     "row " + std::to_string(row) + " out_vector_0");
        check_signed(sign_extend(chip->out_vector_1, 20), vector.expected_out[1],
                     "row " + std::to_string(row) + " out_vector_1");
        check_signed(sign_extend(chip->out_vector_2, 20), vector.expected_out[2],
                     "row " + std::to_string(row) + " out_vector_2");
        check_signed(sign_extend(chip->out_vector_3, 20), vector.expected_out[3],
                     "row " + std::to_string(row) + " out_vector_3");
        check_signed(chip->gf_next_state, vector.expected_gf_state,
                     "row " + std::to_string(row) + " gf_next_state");

        chip->clk = 0;
        chip->eval();
        if (trace) { trace->dump(static_cast<vluint64_t>(tick++)); }
    }

    // Phase B: freeze the pads and run the instruction-set self-test in the core.
    std::cout << "[INFO] phase B, RV32E instruction-set self-test from " << selftest_path
              << std::endl;
    chip->in_vector_0 = static_cast<uint8_t>(selftest["in_vector_0"] & 0xFF);
    chip->in_vector_1 = static_cast<uint8_t>(selftest["in_vector_1"] & 0xFF);
    chip->in_vector_2 = static_cast<uint8_t>(selftest["in_vector_2"] & 0xFF);
    chip->in_vector_3 = static_cast<uint8_t>(selftest["in_vector_3"] & 0xFF);
    chip->gf_symbol_in = static_cast<uint8_t>(selftest["gf_symbol_in"] & 0xFF);
    chip->smt_in_bits = static_cast<uint8_t>(selftest["smt_in_bits"] & 0xFF);
    for (int cycle = 0; cycle < 4; ++cycle) {
        chip->clk = 0;
        chip->eval();
        if (trace) { trace->dump(static_cast<vluint64_t>(tick++)); }
        chip->clk = 1;
        chip->eval();
        if (trace) { trace->dump(static_cast<vluint64_t>(tick++)); }
    }

    check_signed(chip->out_vector_2, sign_extend(selftest["exp_x11"], 20),
                 "frozen pads out_vector_2 before the core starts");

    chip->core_start = 1;
    int cycles = 0;
    while (!chip->core_done && cycles < 400) {
        chip->clk = 1;
        chip->eval();
        if (trace) { trace->dump(static_cast<vluint64_t>(tick++)); }
        chip->clk = 0;
        chip->eval();
        if (trace) { trace->dump(static_cast<vluint64_t>(tick++)); }
        ++cycles;
    }
    std::cout << "[INFO] self-test took " << cycles << " clock cycles" << std::endl;

    check(chip->core_done == 1, "core_done after the self-test", chip->core_done, 1);
    check_signed(chip->core_pass, selftest["exp_core_pass"], "core_pass");
    check_signed(chip->core_illegal_seen, selftest["exp_core_illegal_seen"],
                 "core_illegal_seen (the firmware executes one x16 encoding on purpose)");
    check_signed(chip->dbg_reg_x8, selftest["exp_x8"], "x8 after the SMT readback instruction");
    check_signed(chip->dbg_reg_x10, selftest["exp_x10"], "x10 after the GF step instruction");
    check_signed(chip->dbg_reg_x11, selftest["exp_x11"], "x11 after the mesh row readback");
    check_signed(chip->dbg_reg_x13, selftest["exp_x13"], "x13 after the status readback");
    check_signed(chip->dbg_reg_x15, selftest["exp_x15"],
                 "x15 error accumulator, every firmware instruction check must pass");
    // The status word the firmware read into x13 carries the counter as it stood before
    // that request, because the readback is combinational. The counter register has also
    // counted the status request itself, so the harness compares it with its own key.
    check_signed(chip->dbg_cop_count, selftest["exp_cop_count"],
                 "coprocessor counter, one count per request including the status read");
    check_signed(chip->gf_next_state, selftest["exp_gf_state_final"],
                 "gf_next_state after the self-test, the idle symbol must return the initial state");

    chip->core_start = 0;
    chip->clk = 0;
    chip->eval();

    if (trace) {
        trace->close();
        delete trace;
        std::cout << "[INFO] waveform written to " << vcd_path << std::endl;
    }
    // Verilator 4.034 onwards cleans up on its own, and the explicit final() call is
    // gone in Verilator 5, so the model is simply deleted here.
    delete chip;

    if (g_failure_count > 0) {
        std::cout << "checked " << vectors.size() << " vector rows, " << g_failure_count
                  << " mismatches" << std::endl;
        std::cout << "FINAL VERDICT: FAIL" << std::endl;
        return 1;
    }
    std::cout << "checked " << vectors.size() << " vector rows plus the core self-test, 0 mismatches"
              << std::endl;
    std::cout << "FINAL VERDICT: PASS" << std::endl;
    return 0;
}
