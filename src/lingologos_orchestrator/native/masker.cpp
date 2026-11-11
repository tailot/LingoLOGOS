#include <cmath>
#include <vector>
#include <cstdint>
#include <limits>
#include <cstdlib>
#include <algorithm>

extern "C" {

typedef struct {
    float* mask_buffer;
    uint32_t vocab_size;
    uint32_t current_state;
} VolatileMaskerState;

// Initializes the volatile buffer in contiguous RAM
VolatileMaskerState* masker_init(uint32_t vocab_size) {
    VolatileMaskerState* state = new VolatileMaskerState();
    state->vocab_size = vocab_size;
    state->current_state = 0;
    state->mask_buffer = (float*)malloc(vocab_size * sizeof(float));

    // Initialize all values to -INF
    for (uint32_t i = 0; i < vocab_size; ++i) {
        state->mask_buffer[i] = -std::numeric_limits<float>::infinity();
    }
    return state;
}

// Updates allowed indices without reallocating memory
void masker_update_allowed(VolatileMaskerState* state, const uint32_t* allowed_indices, uint32_t count) {
    if (!state || !state->mask_buffer) return;
    std::fill_n(state->mask_buffer, state->vocab_size, -std::numeric_limits<float>::infinity());

    for (uint32_t i = 0; i < count; ++i) {
        uint32_t idx = allowed_indices[i];
        if (idx < state->vocab_size) {
            state->mask_buffer[idx] = 0.0f; // Enable admissible token
        }
    }
}

// Applies in-place Hadamard addition over model logits
void masker_apply_in_place(VolatileMaskerState* state, float* logits) {
    if (!state || !state->mask_buffer || !logits) return;
    for (uint32_t i = 0; i < state->vocab_size; ++i) {
        logits[i] += state->mask_buffer[i];
    }
}

// Instant memory cleanup upon finishing request
void masker_free(VolatileMaskerState* state) {
    if (state) {
        if (state->mask_buffer) free(state->mask_buffer);
        delete state;
    }
}

}
