#![allow(clippy::needless_borrows_for_generic_args, unused_variables)]
#![cfg(feature = "mlx")]

use mlx_rs::Array;
use std::collections::HashMap;

fn load_codebert_weights() -> Option<HashMap<String, Array>> {
    let home = dirs::home_dir()?;
    let path = home.join(".cache/huggingface/hub/models--microsoft--codebert-base/snapshots/3b0952feddeffad0063f274080e3c23d75e7eb39/model.safetensors");
    if !path.exists() {
        return None;
    }
    Array::load_safetensors(&path).ok()
}

#[test]
fn mlx_embedding_layer() {
    let weights = match load_codebert_weights() {
        Some(w) => w,
        None => {
            eprintln!("Skipping: codebert weights not found");
            return;
        }
    };

    let input_ids = Array::from_slice(&[0i32, 250, 2], &[1, 3]);
    let attention_mask = Array::from_slice(&[1i32, 1, 1], &[1, 3]);

    // RoBERTa position IDs
    let mask_i32 = attention_mask.as_type::<i32>().unwrap();
    let cumsum = mask_i32.cumsum(Some(1), None, None).unwrap();
    let position_ids = &(&cumsum * &mask_i32) + &Array::from_int(1i32);
    position_ids.eval().unwrap();
    let pos_data: &[i32] = position_ids.as_slice();
    eprintln!("Position IDs: {:?}", pos_data);
    assert_eq!(pos_data, &[2, 3, 4]);

    let word_emb_table = weights.get("embeddings.word_embeddings.weight").unwrap();
    let pos_emb_table = weights
        .get("embeddings.position_embeddings.weight")
        .unwrap();
    let type_emb_table = weights
        .get("embeddings.token_type_embeddings.weight")
        .unwrap();

    let ids_i32 = input_ids.as_type::<i32>().unwrap();
    let word_emb = word_emb_table.take_axis(&ids_i32, 0).unwrap();
    let pos_emb = pos_emb_table.take_axis(&position_ids, 0).unwrap();
    let token_type_ids = Array::zeros::<i32>(&[1, 3]).unwrap();
    let type_emb = type_emb_table.take_axis(&token_type_ids, 0).unwrap();

    let combined = &(&word_emb + &pos_emb) + &type_emb;

    let ln_weight = weights.get("embeddings.LayerNorm.weight").unwrap();
    let ln_bias = weights.get("embeddings.LayerNorm.bias").unwrap();
    let normed = mlx_rs::fast::layer_norm(&combined, Some(ln_weight), Some(ln_bias), 1e-5).unwrap();

    normed.eval().unwrap();
    let shape = normed.shape().to_vec();
    eprintln!("Embedding output shape: {:?}", shape);
    assert_eq!(shape, vec![1, 3, 768]);

    let data: &[f32] = normed.as_slice();
    eprintln!("First 5 values: {:?}", &data[..5]);
    let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(norm > 0.0);
    eprintln!("Embedding layer PASSED, norm={norm:.4}");
}

#[test]
fn mlx_self_attention_layer0() {
    let weights = match load_codebert_weights() {
        Some(w) => w,
        None => {
            eprintln!("Skipping: codebert weights not found");
            return;
        }
    };

    // Dummy hidden state [1, 2, 768]
    let hidden = Array::from_slice(&vec![0.01f32; 2 * 768], &[1, 2, 768]);
    let attention_mask = Array::from_slice(&[1i32, 1], &[1, 2]);

    let pfx = "encoder.layer.0";
    let q_w = weights
        .get(&format!("{pfx}.attention.self.query.weight"))
        .unwrap();
    let q_b = weights
        .get(&format!("{pfx}.attention.self.query.bias"))
        .unwrap();
    let k_w = weights
        .get(&format!("{pfx}.attention.self.key.weight"))
        .unwrap();
    let k_b = weights
        .get(&format!("{pfx}.attention.self.key.bias"))
        .unwrap();
    let v_w = weights
        .get(&format!("{pfx}.attention.self.value.weight"))
        .unwrap();
    let v_b = weights
        .get(&format!("{pfx}.attention.self.value.bias"))
        .unwrap();

    // Q, K, V projections
    let q = &hidden.matmul(&q_w.t()).unwrap() + q_b;
    let k = &hidden.matmul(&k_w.t()).unwrap() + k_b;
    let v = &hidden.matmul(&v_w.t()).unwrap() + v_b;

    // Reshape to multi-head
    let q = q
        .reshape(&[1, 2, 12, 64])
        .unwrap()
        .transpose_axes(&[0, 2, 1, 3])
        .unwrap();
    let k = k
        .reshape(&[1, 2, 12, 64])
        .unwrap()
        .transpose_axes(&[0, 2, 1, 3])
        .unwrap();
    let v = v
        .reshape(&[1, 2, 12, 64])
        .unwrap()
        .transpose_axes(&[0, 2, 1, 3])
        .unwrap();

    // Attention scores
    let scale = Array::from_f32(1.0 / 8.0); // sqrt(64) = 8
    let kt = k.transpose_axes(&[0, 1, 3, 2]).unwrap();
    let scores = &q.matmul(&kt).unwrap() * &scale;

    // Mask
    let mask_f32 = attention_mask.as_type::<f32>().unwrap();
    let mask_bias = &(&Array::from_f32(1.0) - &mask_f32) * &Array::from_f32(-1e9);
    let mask_bias = mask_bias.expand_dims(1).unwrap().expand_dims(1).unwrap();
    let scores = &scores + &mask_bias;

    let attn_weights = mlx_rs::ops::softmax_axis(&scores, -1, None).unwrap();
    let attn_output = attn_weights.matmul(&v).unwrap();

    // Reshape back
    let attn_output = attn_output
        .transpose_axes(&[0, 2, 1, 3])
        .unwrap()
        .reshape(&[1, 2, 768])
        .unwrap();

    attn_output.eval().unwrap();
    let shape = attn_output.shape().to_vec();
    eprintln!("Attention output shape: {:?}", shape);
    assert_eq!(shape, vec![1, 2, 768]);

    let data: &[f32] = attn_output.as_slice();
    eprintln!("First 5 attn values: {:?}", &data[..5]);
    eprintln!("Self-attention layer0 PASSED");
}
