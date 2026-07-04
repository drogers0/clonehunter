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

fn linear(x: &Array, weight: &Array, bias: &Array) -> Array {
    &x.matmul(&weight.t()).unwrap() + bias
}

fn layer_norm(x: &Array, weight: &Array, bias: &Array) -> Array {
    mlx_rs::fast::layer_norm(x, Some(weight), Some(bias), 1e-5).unwrap()
}

fn encoder_layer(
    hidden: &Array,
    attention_mask: &Array,
    weights: &HashMap<String, Array>,
    layer_idx: usize,
    batch: i32,
    seq_len: i32,
) -> Array {
    let pfx = format!("encoder.layer.{layer_idx}");

    // Self-attention
    let q = linear(
        hidden,
        weights
            .get(&format!("{pfx}.attention.self.query.weight"))
            .unwrap(),
        weights
            .get(&format!("{pfx}.attention.self.query.bias"))
            .unwrap(),
    );
    let k = linear(
        hidden,
        weights
            .get(&format!("{pfx}.attention.self.key.weight"))
            .unwrap(),
        weights
            .get(&format!("{pfx}.attention.self.key.bias"))
            .unwrap(),
    );
    let v = linear(
        hidden,
        weights
            .get(&format!("{pfx}.attention.self.value.weight"))
            .unwrap(),
        weights
            .get(&format!("{pfx}.attention.self.value.bias"))
            .unwrap(),
    );

    let head_shape = &[batch, seq_len, 12, 64];
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

    let scale = Array::from_f32(1.0 / 8.0);
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
        .reshape(&[batch, seq_len, 768])
        .unwrap();

    // Output projection + residual + LN
    let projected = linear(
        &attn_output,
        weights
            .get(&format!("{pfx}.attention.output.dense.weight"))
            .unwrap(),
        weights
            .get(&format!("{pfx}.attention.output.dense.bias"))
            .unwrap(),
    );
    let residual = &projected + hidden;
    let normed = layer_norm(
        &residual,
        weights
            .get(&format!("{pfx}.attention.output.LayerNorm.weight"))
            .unwrap(),
        weights
            .get(&format!("{pfx}.attention.output.LayerNorm.bias"))
            .unwrap(),
    );

    // FFN
    let intermediate = linear(
        &normed,
        weights
            .get(&format!("{pfx}.intermediate.dense.weight"))
            .unwrap(),
        weights
            .get(&format!("{pfx}.intermediate.dense.bias"))
            .unwrap(),
    );
    let activated = mlx_rs::nn::gelu(&intermediate).unwrap();
    let output = linear(
        &activated,
        weights.get(&format!("{pfx}.output.dense.weight")).unwrap(),
        weights.get(&format!("{pfx}.output.dense.bias")).unwrap(),
    );
    let residual = &output + &normed;
    layer_norm(
        &residual,
        weights
            .get(&format!("{pfx}.output.LayerNorm.weight"))
            .unwrap(),
        weights
            .get(&format!("{pfx}.output.LayerNorm.bias"))
            .unwrap(),
    )
}

#[test]
fn mlx_full_12_layers() {
    let weights = match load_codebert_weights() {
        Some(w) => w,
        None => {
            eprintln!("Skipping: codebert weights not found");
            return;
        }
    };

    // Simple input: one token
    let input_ids = Array::from_slice(&[0i32, 250], &[1, 2]);
    let attention_mask = Array::from_slice(&[1i32, 1], &[1, 2]);

    // Embedding layer
    let mask_i32 = attention_mask.as_type::<i32>().unwrap();
    let cumsum = mask_i32.cumsum(Some(1), None, None).unwrap();
    let position_ids = &(&cumsum * &mask_i32) + &Array::from_int(1i32);

    let word_emb = weights
        .get("embeddings.word_embeddings.weight")
        .unwrap()
        .take_axis(&input_ids.as_type::<i32>().unwrap(), 0)
        .unwrap();
    let pos_emb = weights
        .get("embeddings.position_embeddings.weight")
        .unwrap()
        .take_axis(&position_ids, 0)
        .unwrap();
    let token_type_ids = Array::zeros::<i32>(&[1, 2]).unwrap();
    let type_emb = weights
        .get("embeddings.token_type_embeddings.weight")
        .unwrap()
        .take_axis(&token_type_ids, 0)
        .unwrap();

    let combined = &(&word_emb + &pos_emb) + &type_emb;
    let mut hidden = layer_norm(
        &combined,
        weights.get("embeddings.LayerNorm.weight").unwrap(),
        weights.get("embeddings.LayerNorm.bias").unwrap(),
    );

    eprintln!("Embedding done");

    // Run through all 12 layers
    for i in 0..12 {
        hidden = encoder_layer(&hidden, &attention_mask, &weights, i, 1, 2);
        // Eval after each layer to force computation and catch errors early
        hidden.eval().unwrap();
        let data: &[f32] = hidden.as_slice();
        eprintln!(
            "Layer {i} done, first 3: [{:.6}, {:.6}, {:.6}]",
            data[0], data[1], data[2]
        );
    }

    let shape = hidden.shape().to_vec();
    assert_eq!(shape, vec![1, 2, 768]);
    eprintln!("Full 12-layer forward PASSED");
}
