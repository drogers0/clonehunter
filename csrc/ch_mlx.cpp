/* CloneHunter MLX shim — RoBERTa (codebert-base) forward pass over Apple mlx-c.
 *
 * Ports the exact op sequence of the former mlx-rs backend (src/embedding/
 * mlx_backend.rs history) op-for-op onto the mlx-c C API, so the frozen
 * detection baseline (578 findings on click; scores within 1e-4) is preserved.
 *
 * Memory: every mlx op returns an OWNED mlx_array via an out-param; the `Arr`
 * RAII type default-constructs an empty handle, passes &a to the op (which
 * replaces it), and frees on scope exit. Weight lookups likewise return an
 * owned copy of the array stored in the retained safetensors map — free them
 * the same way (the map keeps its own reference).
 *
 * Errors: MLX's default error handler calls exit(-1). We install a custom
 * handler that records the message into a thread-local instead, so an op error
 * surfaces as a nonzero return (Rust maps it to EmbeddingError) rather than
 * killing the whole scan.
 */
#include "ch_mlx.h"

#include "mlx/c/mlx.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ── Model constants (codebert-base / RoBERTa-base) ───────────────────────────
constexpr int NUM_LAYERS = 12;
constexpr int HIDDEN_SIZE = 768;
constexpr int NUM_HEADS = 12;
constexpr int HEAD_DIM = HIDDEN_SIZE / NUM_HEADS; // 64
constexpr float LAYER_NORM_EPS = 1e-5f;
constexpr int PAD_TOKEN_ID = 1;

// ── Thread-local last error + non-fatal MLX error handler ────────────────────
thread_local std::string g_err;

void ch_error_handler(const char* msg, void* /*data*/) {
    // Replace MLX's default exit(-1): just record the message. The op that
    // triggered it returns nonzero, which our CK() macro turns into a throw.
    g_err = msg ? msg : "unknown MLX error";
}

void install_error_handler() {
    // Global, idempotent — safe to call on every entry point.
    //
    // Note: the handler slot is process-global but g_err is thread_local. This is
    // consistent only because MLX catches and re-throws op errors on the *calling*
    // thread (e.g. mlx_array_eval), so the handler runs on the same thread that then
    // reads g_err via CK()/ch_mlx_last_error(). Async/worker-thread failures (e.g. a
    // Metal command-buffer assertion) bypass this path entirely and abort — which is
    // why embed() must stay single-threaded per ctx.
    mlx_set_error_handler(ch_error_handler, nullptr, nullptr);
}

struct MlxError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Throw on a nonzero mlx return code, preferring the handler-recorded message.
#define CK(expr, what)                                                       \
    do {                                                                     \
        if ((expr) != 0)                                                     \
            throw MlxError(g_err.empty() ? std::string(what) : g_err);       \
    } while (0)

// ── RAII wrapper over mlx_array ──────────────────────────────────────────────
struct Arr {
    mlx_array a;
    Arr() : a(mlx_array_new()) {}
    explicit Arr(mlx_array x) : a(x) {}
    ~Arr() { mlx_array_free(a); }
    Arr(const Arr&) = delete;
    Arr& operator=(const Arr&) = delete;
    Arr(Arr&& o) noexcept : a(o.a) { o.a = mlx_array_new(); }
    Arr& operator=(Arr&& o) noexcept {
        if (this != &o) {
            mlx_array_free(a);
            a = o.a;
            o.a = mlx_array_new();
        }
        return *this;
    }
};

// ── Scalar constructors ──────────────────────────────────────────────────────
Arr f32(float v) { return Arr(mlx_array_new_float32(v)); }
Arr i32(int v) { return Arr(mlx_array_new_int(v)); }

// ── Op helpers (each threads the caller's stream) ────────────────────────────
Arr add(const Arr& a, const Arr& b, mlx_stream s) {
    Arr r;
    CK(mlx_add(&r.a, a.a, b.a, s), "add");
    return r;
}
Arr mul(const Arr& a, const Arr& b, mlx_stream s) {
    Arr r;
    CK(mlx_multiply(&r.a, a.a, b.a, s), "multiply");
    return r;
}
Arr sub(const Arr& a, const Arr& b, mlx_stream s) {
    Arr r;
    CK(mlx_subtract(&r.a, a.a, b.a, s), "subtract");
    return r;
}
Arr divide(const Arr& a, const Arr& b, mlx_stream s) {
    Arr r;
    CK(mlx_divide(&r.a, a.a, b.a, s), "divide");
    return r;
}
Arr maximum(const Arr& a, const Arr& b, mlx_stream s) {
    Arr r;
    CK(mlx_maximum(&r.a, a.a, b.a, s), "maximum");
    return r;
}
Arr matmul(const Arr& a, const Arr& b, mlx_stream s) {
    Arr r;
    CK(mlx_matmul(&r.a, a.a, b.a, s), "matmul");
    return r;
}
Arr erf(const Arr& a, mlx_stream s) {
    Arr r;
    CK(mlx_erf(&r.a, a.a, s), "erf");
    return r;
}
Arr astype(const Arr& a, mlx_dtype dt, mlx_stream s) {
    Arr r;
    CK(mlx_astype(&r.a, a.a, dt, s), "astype");
    return r;
}
Arr take_axis(const Arr& a, const Arr& idx, int axis, mlx_stream s) {
    Arr r;
    CK(mlx_take_axis(&r.a, a.a, idx.a, axis, s), "take_axis");
    return r;
}
// Matches mlx-rs softmax_axis(_, _, None): precise = false.
Arr softmax_axis(const Arr& a, int axis, mlx_stream s) {
    Arr r;
    CK(mlx_softmax_axis(&r.a, a.a, axis, /*precise=*/false, s), "softmax");
    return r;
}
// Matches mlx-rs cumsum(Some(axis), None, None): reverse = false, inclusive = true.
Arr cumsum(const Arr& a, int axis, mlx_stream s) {
    Arr r;
    CK(mlx_cumsum(&r.a, a.a, axis, /*reverse=*/false, /*inclusive=*/true, s),
       "cumsum");
    return r;
}
// Matches mlx-rs sum_axis(axis, None): keepdims = false.
Arr sum_axis(const Arr& a, int axis, mlx_stream s) {
    Arr r;
    CK(mlx_sum_axis(&r.a, a.a, axis, /*keepdims=*/false, s), "sum_axis");
    return r;
}
Arr expand_dims(const Arr& a, int axis, mlx_stream s) {
    Arr r;
    CK(mlx_expand_dims(&r.a, a.a, axis, s), "expand_dims");
    return r;
}
Arr reshape(const Arr& a, const std::vector<int>& shape, mlx_stream s) {
    Arr r;
    CK(mlx_reshape(&r.a, a.a, shape.data(), shape.size(), s), "reshape");
    return r;
}
Arr transpose_axes(const Arr& a, const std::vector<int>& axes, mlx_stream s) {
    Arr r;
    CK(mlx_transpose_axes(&r.a, a.a, axes.data(), axes.size(), s), "transpose");
    return r;
}
Arr layer_norm(const Arr& x, const Arr& w, const Arr& b, mlx_stream s) {
    Arr r;
    CK(mlx_fast_layer_norm(&r.a, x.a, w.a, b.a, LAYER_NORM_EPS, s), "layer_norm");
    return r;
}
Arr zeros_i32(int batch, int seq, mlx_stream s) {
    Arr r;
    const int shape[2] = {batch, seq};
    CK(mlx_zeros(&r.a, shape, 2, MLX_INT32, s), "zeros");
    return r;
}

} // namespace

// ── Context: retained weights map + compute stream ───────────────────────────
struct ch_mlx_ctx {
    mlx_map_string_to_array weights;
    mlx_stream stream;
};

namespace {

// Fetch a weight by key as an OWNED array (the map keeps its own reference).
// rc == 2 means the key is absent (mlx_error is NOT called for that path).
Arr weight(mlx_map_string_to_array weights, const std::string& key) {
    Arr r;
    int rc = mlx_map_string_to_array_get(&r.a, weights, key.c_str());
    if (rc == 2)
        throw MlxError("missing weight: " + key);
    CK(rc, "weight get: " + key);
    return r;
}

// linear(x) = x @ wt + bias, where `wt` is the [in, out] weight (already transposed
// once at load — see prepare_linear_weights — so no per-forward transpose/materialize).
Arr linear(const Arr& x, const Arr& wt, const Arr& b, mlx_stream s) {
    return add(matmul(x, wt, s), b, s);
}

// Exact-erf GELU, matching mlx-rs nn::gelu: x * (1 + erf(x / sqrt(2))) / 2.
// Divide by sqrt(2) (not multiply by a reciprocal); the final /2 is exact.
Arr gelu(const Arr& x, mlx_stream s) {
    Arr e = erf(divide(x, f32(std::sqrt(2.0f)), s), s);
    return divide(mul(x, add(f32(1.0f), e, s), s), f32(2.0f), s);
}

// RoBERTa embeddings: word + position + token_type, then LayerNorm.
// position_ids = cumsum(mask, axis=1) * mask + padding_idx.
Arr roberta_embeddings(
    const Arr& input_ids,
    const Arr& attention_mask,
    mlx_map_string_to_array w,
    int batch,
    int seq,
    mlx_stream s) {
    Arr mask_i32 = astype(attention_mask, MLX_INT32, s);
    Arr position_ids =
        add(mul(cumsum(mask_i32, 1, s), mask_i32, s), i32(PAD_TOKEN_ID), s);
    Arr token_type_ids = zeros_i32(batch, seq, s);
    Arr ids_i32 = astype(input_ids, MLX_INT32, s);

    Arr word = take_axis(weight(w, "embeddings.word_embeddings.weight"), ids_i32, 0, s);
    Arr pos =
        take_axis(weight(w, "embeddings.position_embeddings.weight"), position_ids, 0, s);
    Arr typ = take_axis(
        weight(w, "embeddings.token_type_embeddings.weight"), token_type_ids, 0, s);

    Arr combined = add(add(word, pos, s), typ, s);
    return layer_norm(
        combined,
        weight(w, "embeddings.LayerNorm.weight"),
        weight(w, "embeddings.LayerNorm.bias"),
        s);
}

Arr encoder_layer(
    const Arr& hidden,
    const Arr& mask_bias, // precomputed once per forward (constant across layers)
    mlx_map_string_to_array w,
    int layer_idx,
    int batch,
    int seq,
    mlx_stream s) {
    const std::string p = "encoder.layer." + std::to_string(layer_idx);
    auto W = [&](const char* suffix) { return weight(w, p + suffix); };

    // Q, K, V projections.
    Arr q = linear(hidden, W(".attention.self.query.weight"), W(".attention.self.query.bias"), s);
    Arr k = linear(hidden, W(".attention.self.key.weight"), W(".attention.self.key.bias"), s);
    Arr v = linear(hidden, W(".attention.self.value.weight"), W(".attention.self.value.bias"), s);

    // [batch, seq, hidden] → [batch, heads, seq, head_dim].
    const std::vector<int> head_shape = {batch, seq, NUM_HEADS, HEAD_DIM};
    q = transpose_axes(reshape(q, head_shape, s), {0, 2, 1, 3}, s);
    k = transpose_axes(reshape(k, head_shape, s), {0, 2, 1, 3}, s);
    v = transpose_axes(reshape(v, head_shape, s), {0, 2, 1, 3}, s);

    // scores = (Q @ Kᵀ) / sqrt(head_dim).
    Arr kt = transpose_axes(k, {0, 1, 3, 2}, s);
    Arr scores = mul(matmul(q, kt, s), f32(1.0f / std::sqrt((float)HEAD_DIM)), s);

    scores = add(scores, mask_bias, s);

    Arr attn = matmul(softmax_axis(scores, -1, s), v, s);

    // [batch, heads, seq, head_dim] → [batch, seq, hidden].
    attn = reshape(
        transpose_axes(attn, {0, 2, 1, 3}, s), {batch, seq, HIDDEN_SIZE}, s);

    // Output projection + residual + LayerNorm.
    Arr projected =
        linear(attn, W(".attention.output.dense.weight"), W(".attention.output.dense.bias"), s);
    Arr normed = layer_norm(
        add(projected, hidden, s),
        W(".attention.output.LayerNorm.weight"),
        W(".attention.output.LayerNorm.bias"),
        s);

    // FFN: Linear → GELU → Linear + residual + LayerNorm.
    Arr intermediate =
        linear(normed, W(".intermediate.dense.weight"), W(".intermediate.dense.bias"), s);
    Arr output = linear(gelu(intermediate, s), W(".output.dense.weight"), W(".output.dense.bias"), s);
    return layer_norm(
        add(output, normed, s),
        W(".output.LayerNorm.weight"),
        W(".output.LayerNorm.bias"),
        s);
}

// Mean-pool over the sequence axis, weighted by the (already-f32) attention mask.
Arr mean_pool(const Arr& hidden, const Arr& mask_f32, mlx_stream s) {
    Arr mask_3d = expand_dims(mask_f32, -1, s);
    Arr summed = sum_axis(mul(hidden, mask_3d, s), 1, s);
    Arr counts = maximum(sum_axis(mask_3d, 1, s), f32(1.0f), s);
    return divide(summed, counts, s);
}

// Pre-transpose (and materialize once) each linear-layer weight in place. The forward
// pass then consumes an already-transposed [in, out] weight instead of re-transposing and
// re-materializing it on every batch — a large win over 212 batches × 72 linear ops.
// Numerically identical: transpose-then-matmul == matmul-of-transposed-view.
void prepare_linear_weights(mlx_map_string_to_array w, mlx_stream s) {
    static const char* const suffixes[] = {
        ".attention.self.query.weight",
        ".attention.self.key.weight",
        ".attention.self.value.weight",
        ".attention.output.dense.weight",
        ".intermediate.dense.weight",
        ".output.dense.weight",
    };
    for (int i = 0; i < NUM_LAYERS; ++i) {
        const std::string p = "encoder.layer." + std::to_string(i);
        for (const char* suffix : suffixes) {
            const std::string key = p + suffix;
            Arr wt = transpose_axes(weight(w, key), {1, 0}, s);
            CK(mlx_array_eval(wt.a), "eval linear weight");
            CK(mlx_map_string_to_array_insert(w, key.c_str(), wt.a), "insert linear weight");
        }
    }
}

// Validate that a representative set of weights loaded (fail fast at load time).
bool validate_weights(mlx_map_string_to_array w, std::string& err) {
    const char* required[] = {
        "embeddings.word_embeddings.weight",
        "encoder.layer.0.attention.self.query.weight",
        "encoder.layer.11.output.LayerNorm.weight",
    };
    for (const char* key : required) {
        Arr probe;
        int rc = mlx_map_string_to_array_get(&probe.a, w, key);
        if (rc != 0) {
            err = (rc == 2) ? std::string("missing weight: ") + key
                  : g_err.empty() ? std::string("weight get failed: ") + key
                                  : g_err;
            return false;
        }
    }
    return true;
}

} // namespace

// ── C ABI ────────────────────────────────────────────────────────────────────
extern "C" ch_mlx_ctx* ch_mlx_ctx_new(const char* safetensors_path) {
    install_error_handler();
    g_err.clear();
    try {
        bool metal = false;
        mlx_metal_is_available(&metal);
        mlx_stream stream =
            metal ? mlx_default_gpu_stream_new() : mlx_default_cpu_stream_new();

        // safetensors loading must run on a CPU stream (mlx enforces this). Use a
        // dedicated CPU stream when computing on GPU; reuse the compute stream when
        // it is already CPU (avoids a second handle to the same default stream).
        // The loaded arrays are device-agnostic — the GPU forward pass moves them
        // as needed.
        mlx_stream load_stream = metal ? mlx_default_cpu_stream_new() : stream;

        // Both out-maps must be pre-constructed (load dereferences them).
        mlx_map_string_to_array weights = mlx_map_string_to_array_new();
        mlx_map_string_to_string metadata = mlx_map_string_to_string_new();
        int rc = mlx_load_safetensors(&weights, &metadata, safetensors_path, load_stream);
        mlx_map_string_to_string_free(metadata);
        if (metal)
            mlx_stream_free(load_stream);

        std::string err;
        if (rc != 0) {
            err = g_err.empty()
                ? std::string("load_safetensors failed: ") + safetensors_path
                : g_err;
        } else if (!validate_weights(weights, err)) {
            // err set by validate_weights
        } else {
            try {
                prepare_linear_weights(weights, stream);
                return new ch_mlx_ctx{weights, stream};
            } catch (const std::exception& e) {
                err = e.what();
            }
        }

        mlx_map_string_to_array_free(weights);
        mlx_stream_free(stream);
        g_err = err;
        return nullptr;
    } catch (const std::exception& e) {
        g_err = e.what();
        return nullptr;
    }
}

extern "C" int ch_mlx_embed(
    ch_mlx_ctx* ctx,
    const int32_t* ids,
    const int32_t* mask,
    int batch,
    int seq,
    float* out) {
    install_error_handler();
    g_err.clear();
    if (!ctx) {
        g_err = "null ctx";
        return 1;
    }
    if (batch == 0)
        return 0;
    try {
        const mlx_stream s = ctx->stream;
        const int shape[2] = {batch, seq};
        Arr input_ids(mlx_array_new_data(ids, shape, 2, MLX_INT32));
        Arr attention_mask(mlx_array_new_data(mask, shape, 2, MLX_INT32));

        // Mask-derived tensors are constant across the 12 layers — build once, not per layer.
        Arr mask_f32 = astype(attention_mask, MLX_FLOAT32, s);
        Arr mask_bias = expand_dims(
            expand_dims(mul(sub(f32(1.0f), mask_f32, s), f32(-1e9f), s), 1, s), 1, s);

        Arr hidden =
            roberta_embeddings(input_ids, attention_mask, ctx->weights, batch, seq, s);
        for (int i = 0; i < NUM_LAYERS; ++i)
            hidden = encoder_layer(hidden, mask_bias, ctx->weights, i, batch, seq, s);
        Arr pooled = mean_pool(hidden, mask_f32, s);

        CK(mlx_array_eval(pooled.a), "eval");
        const float* data = mlx_array_data_float32(pooled.a);
        if (!data || mlx_array_size(pooled.a) != (size_t)batch * HIDDEN_SIZE) {
            g_err = "pooled output was null / not f32 / wrong size";
            return 1;
        }
        std::memcpy(out, data, (size_t)batch * HIDDEN_SIZE * sizeof(float));
        return 0;
    } catch (const std::exception& e) {
        g_err = e.what();
        return 1;
    }
}

extern "C" void ch_mlx_ctx_free(ch_mlx_ctx* ctx) {
    if (!ctx)
        return;
    mlx_map_string_to_array_free(ctx->weights);
    mlx_stream_free(ctx->stream);
    delete ctx;
}

extern "C" bool ch_mlx_metal_available(void) {
    // This is the FIRST MLX call the process makes (MlxEmbedder::new probes it before
    // ch_mlx_ctx_new), so install the non-fatal error handler here too — otherwise a
    // throwing metal probe would hit MLX's default exit(-1) and crash the scan.
    install_error_handler();
    bool r = false;
    mlx_metal_is_available(&r);
    return r;
}

extern "C" int ch_mlx_selftest(void) {
    install_error_handler();
    g_err.clear();
    try {
        bool metal = false;
        mlx_metal_is_available(&metal);
        mlx_stream s =
            metal ? mlx_default_gpu_stream_new() : mlx_default_cpu_stream_new();
        const float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        const int shape[2] = {2, 2};
        Arr a(mlx_array_new_data(data, shape, 2, MLX_FLOAT32));
        Arr r = matmul(a, a, s);
        int rc = mlx_array_eval(r.a);
        mlx_stream_free(s);
        if (rc != 0) {
            if (g_err.empty())
                g_err = "selftest eval failed";
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        g_err = e.what();
        return 1;
    }
}

extern "C" const char* ch_mlx_last_error(void) {
    return g_err.empty() ? nullptr : g_err.c_str();
}
