#![allow(clippy::needless_borrows_for_generic_args, dead_code, unused_variables)]
#![cfg(feature = "mlx")]

//! Numeric parity test: MLX embeddings vs PyTorch CPU reference.
//! Run: `cargo test --features mlx --test mlx_parity -- --nocapture --test-threads=1`

use mlx_rs::Array;
use std::collections::HashMap;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

const CODEBERT_TOKENIZER_BYTES: &[u8] = include_bytes!("../src/embedding/tokenizer.json");
const HIDDEN_SIZE: usize = 768;
const NUM_LAYERS: usize = 12;
const NUM_HEADS: usize = 12;
const HEAD_DIM: usize = 64;
const LAYER_NORM_EPS: f32 = 1e-5;
const PAD_TOKEN_ID: i32 = 1;

fn ie(e: impl std::fmt::Display) -> String {
    format!("{e}")
}

fn linear(x: &Array, weight: &Array, bias: &Array) -> Array {
    &x.matmul(&weight.t()).unwrap() + bias
}

fn layer_norm(x: &Array, weight: &Array, bias: &Array) -> Array {
    mlx_rs::fast::layer_norm(x, Some(weight), Some(bias), LAYER_NORM_EPS).unwrap()
}

fn roberta_embeddings(
    input_ids: &Array,
    attention_mask: &Array,
    weights: &HashMap<String, Array>,
) -> Array {
    let mask_i32 = attention_mask.as_type::<i32>().unwrap();
    let cumsum = mask_i32.cumsum(Some(1), None, None).unwrap();
    let position_ids = &(&cumsum * &mask_i32) + &Array::from_int(PAD_TOKEN_ID);
    let shape = input_ids.shape().to_vec();
    let token_type_ids = Array::zeros::<i32>(&shape).unwrap();

    let ids_i32 = input_ids.as_type::<i32>().unwrap();
    let word_emb = weights["embeddings.word_embeddings.weight"]
        .take_axis(&ids_i32, 0)
        .unwrap();
    let pos_emb = weights["embeddings.position_embeddings.weight"]
        .take_axis(&position_ids, 0)
        .unwrap();
    let type_emb = weights["embeddings.token_type_embeddings.weight"]
        .take_axis(&token_type_ids, 0)
        .unwrap();

    let combined = &(&word_emb + &pos_emb) + &type_emb;
    layer_norm(
        &combined,
        &weights["embeddings.LayerNorm.weight"],
        &weights["embeddings.LayerNorm.bias"],
    )
}

fn encoder_layer(
    hidden: &Array,
    attention_mask: &Array,
    weights: &HashMap<String, Array>,
    i: usize,
    batch: i32,
    seq_len: i32,
) -> Array {
    let pfx = format!("encoder.layer.{i}");
    let w = |suffix: &str| -> &Array { &weights[&format!("{pfx}.{suffix}")] };

    let q = linear(
        hidden,
        w("attention.self.query.weight"),
        w("attention.self.query.bias"),
    );
    let k = linear(
        hidden,
        w("attention.self.key.weight"),
        w("attention.self.key.bias"),
    );
    let v = linear(
        hidden,
        w("attention.self.value.weight"),
        w("attention.self.value.bias"),
    );

    let head_shape = &[batch, seq_len, NUM_HEADS as i32, HEAD_DIM as i32];
    let q = q
        .reshape(head_shape)
        .unwrap()
        .transpose_axes(&[0, 2, 1, 3])
        .unwrap();
    let k = k
        .reshape(head_shape)
        .unwrap()
        .transpose_axes(&[0, 2, 1, 3])
        .unwrap();
    let v = v
        .reshape(head_shape)
        .unwrap()
        .transpose_axes(&[0, 2, 1, 3])
        .unwrap();

    let scale = Array::from_f32(1.0 / (HEAD_DIM as f32).sqrt());
    let kt = k.transpose_axes(&[0, 1, 3, 2]).unwrap();
    let scores = &q.matmul(&kt).unwrap() * &scale;

    let mask_f32 = attention_mask.as_type::<f32>().unwrap();
    let mask_bias = &(&Array::from_f32(1.0) - &mask_f32) * &Array::from_f32(-1e9);
    let mask_bias = mask_bias.expand_dims(1).unwrap().expand_dims(1).unwrap();
    let scores = &scores + &mask_bias;
    let attn_weights = mlx_rs::ops::softmax_axis(&scores, -1, None).unwrap();
    let attn_output = attn_weights.matmul(&v).unwrap();
    let attn_output = attn_output
        .transpose_axes(&[0, 2, 1, 3])
        .unwrap()
        .reshape(&[batch, seq_len, HIDDEN_SIZE as i32])
        .unwrap();

    let projected = linear(
        &attn_output,
        w("attention.output.dense.weight"),
        w("attention.output.dense.bias"),
    );
    let residual = &projected + hidden;
    let normed = layer_norm(
        &residual,
        w("attention.output.LayerNorm.weight"),
        w("attention.output.LayerNorm.bias"),
    );

    let intermediate = linear(
        &normed,
        w("intermediate.dense.weight"),
        w("intermediate.dense.bias"),
    );
    let activated = mlx_rs::nn::gelu(&intermediate).unwrap();
    let output = linear(&activated, w("output.dense.weight"), w("output.dense.bias"));
    let residual = &output + &normed;
    layer_norm(
        &residual,
        w("output.LayerNorm.weight"),
        w("output.LayerNorm.bias"),
    )
}

fn embed_single(text: &str, tokenizer: &Tokenizer, weights: &HashMap<String, Array>) -> Vec<f32> {
    let encoding = tokenizer.encode(text, true).unwrap();
    let ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
    let mask: Vec<i32> = encoding
        .get_attention_mask()
        .iter()
        .map(|&m| m as i32)
        .collect();
    let seq_len = ids.len() as i32;

    let input_ids = Array::from_slice(&ids, &[1, seq_len]);
    let attention_mask = Array::from_slice(&mask, &[1, seq_len]);

    let mut hidden = roberta_embeddings(&input_ids, &attention_mask, weights);
    for i in 0..NUM_LAYERS {
        hidden = encoder_layer(&hidden, &attention_mask, weights, i, 1, seq_len);
    }

    // Mean pool
    let mask_f32 = attention_mask.as_type::<f32>().unwrap();
    let mask_3d = mask_f32.expand_dims(-1).unwrap();
    let masked = &hidden * &mask_3d;
    let summed = masked.sum_axis(1, None).unwrap();
    let counts = mask_3d.sum_axis(1, None).unwrap();
    let counts = mlx_rs::ops::maximum(&counts, Array::from_f32(1.0)).unwrap();
    let pooled = &summed / &counts;

    pooled.eval().unwrap();
    let data: &[f32] = pooled.as_slice();
    data.to_vec()
}

fn cosine_sim(a: &[f32], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x as f64 * y).sum();
    let na: f64 = a
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = b.iter().map(|&y| y * y).sum::<f64>().sqrt();
    dot / (na * nb)
}

#[derive(serde::Deserialize)]
struct FixtureEntry {
    text: String,
    embedding: Vec<f64>,
}

#[test]
fn mlx_parity_vs_pytorch() {
    // Load model weights
    let home = dirs::home_dir().unwrap();
    let weights_path = home.join(".cache/huggingface/hub/models--microsoft--codebert-base/snapshots/3b0952feddeffad0063f274080e3c23d75e7eb39/model.safetensors");
    if !weights_path.exists() {
        eprintln!("Skipping: codebert weights not found");
        return;
    }

    let weights = Array::load_safetensors(&weights_path).unwrap();

    // Load tokenizer
    let mut tokenizer = Tokenizer::from_bytes(CODEBERT_TOKENIZER_BYTES).unwrap();
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        pad_id: 1,
        pad_token: "<pad>".into(),
        ..Default::default()
    }));
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: 256,
            ..Default::default()
        }))
        .unwrap();

    // Load PyTorch reference embeddings
    let fixture_path = std::path::Path::new("spike/fixtures/python_embeddings.json");
    let fixture_data = std::fs::read_to_string(fixture_path).unwrap();
    let fixtures: Vec<FixtureEntry> = serde_json::from_str(&fixture_data).unwrap();

    eprintln!("Loaded {} fixture entries", fixtures.len());

    let mut max_cosine_diff: f64 = 0.0;
    let mut max_abs_diff: f64 = 0.0;
    let mut total = 0;

    // Test first N entries (full 205 is too slow in debug mode)
    let n = std::env::var("MLX_PARITY_COUNT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);

    for (i, entry) in fixtures.iter().take(n).enumerate() {
        let mlx_emb = embed_single(&entry.text, &tokenizer, &weights);

        // Cosine similarity
        let cos = cosine_sim(&mlx_emb, &entry.embedding);
        let cosine_diff = (1.0 - cos).abs();

        // Max absolute per-dimension difference
        let abs_diff: f64 = mlx_emb
            .iter()
            .zip(entry.embedding.iter())
            .map(|(&a, &b)| (a as f64 - b).abs())
            .fold(0.0f64, f64::max);

        if cosine_diff > max_cosine_diff {
            max_cosine_diff = cosine_diff;
        }
        if abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
        }
        total += 1;

        eprintln!(
            "  [{i:3}] cosine={cos:.8} diff={cosine_diff:.2e} abs_diff={abs_diff:.2e} text={:.40}",
            entry.text
        );
    }

    eprintln!();
    eprintln!("=== MLX Parity Summary ({total} entries) ===");
    eprintln!("  Max cosine diff:  {max_cosine_diff:.2e}");
    eprintln!("  Max abs diff:     {max_abs_diff:.2e}");
    eprintln!("  (candle ref:      2.68e-6 cosine, ort ref: 2.38e-7)");

    // Determinism check: re-embed first entry
    let emb1 = embed_single(&fixtures[0].text, &tokenizer, &weights);
    let emb2 = embed_single(&fixtures[0].text, &tokenizer, &weights);
    assert_eq!(emb1, emb2, "MLX must be deterministic");
    eprintln!("  Determinism:      PASS");
}
