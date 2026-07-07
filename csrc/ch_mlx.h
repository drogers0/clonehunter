/* CloneHunter MLX shim — C ABI over Apple's mlx-c.
 *
 * The entire RoBERTa (codebert-base) forward pass + mean-pool runs in C++
 * (ch_mlx.cpp) against mlx-c; this header is the only surface the Rust side
 * (src/embedding/mlx_backend.rs) binds. Keep it tiny and stable.
 */
#ifndef CH_MLX_H
#define CH_MLX_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ch_mlx_ctx ch_mlx_ctx;

/* Load codebert-base weights from a safetensors file and pick a compute stream
 * (Metal GPU if available, else CPU). Returns NULL on failure — call
 * ch_mlx_last_error() for the message. */
ch_mlx_ctx* ch_mlx_ctx_new(const char* safetensors_path);

/* Full RoBERTa forward + mean-pool over a padded batch.
 *   ids, mask: row-major [batch*seq] int32 (mask is 0/1); both are copied.
 *   out:       caller-allocated [batch*768] float32.
 * Returns 0 on success, nonzero on error (see ch_mlx_last_error()). */
int ch_mlx_embed(
    ch_mlx_ctx* ctx,
    const int32_t* ids,
    const int32_t* mask,
    int batch,
    int seq,
    float* out);

void ch_mlx_ctx_free(ch_mlx_ctx* ctx);

/* True if a Metal GPU is available. */
bool ch_mlx_metal_available(void);

/* Dylib-only self-test: builds a tiny array, runs one matmul, evals. Returns 0
 * on success. Needs NO weights and NO network — used by the non-ignored smoke
 * test to prove the link + a real op path work. */
int ch_mlx_selftest(void);

/* Thread-local last-error message; valid until the next shim call on this
 * thread. NULL if there is no error. */
const char* ch_mlx_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* CH_MLX_H */
