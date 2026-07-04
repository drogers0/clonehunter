#![cfg(feature = "mlx")]

#[test]
fn mlx_basic_array_ops() {
    use mlx_rs::Array;

    // Basic array creation
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
    eprintln!("Created array, shape: {:?}", a.shape());

    let b = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2]);
    let c = &a + &b;
    c.eval().unwrap();
    let data: &[f32] = c.as_slice();
    eprintln!("a + b = {:?}", data);
    assert_eq!(data, &[6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn mlx_matmul() {
    use mlx_rs::Array;

    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2]);
    let c = a.matmul(&b).unwrap();
    c.eval().unwrap();
    let data: &[f32] = c.as_slice();
    eprintln!("matmul = {:?}", data);
    // [1*5+2*7, 1*6+2*8, 3*5+4*7, 3*6+4*8] = [19, 22, 43, 50]
    assert_eq!(data, &[19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn mlx_safetensors_load() {
    use mlx_rs::Array;

    // Find the codebert safetensors file
    let home = dirs::home_dir().unwrap();
    let path = home
        .join(".cache/huggingface/hub/models--microsoft--codebert-base/snapshots/3b0952feddeffad0063f274080e3c23d75e7eb39/model.safetensors");

    if !path.exists() {
        eprintln!("Skipping: safetensors file not found at {:?}", path);
        return;
    }

    eprintln!("Loading safetensors from {:?}", path);
    let weights = Array::load_safetensors(&path).unwrap();
    eprintln!("Loaded {} weight tensors", weights.len());

    // Check a known weight
    let word_emb = weights.get("embeddings.word_embeddings.weight").unwrap();
    eprintln!(
        "word_embeddings shape: {:?}, dtype: {:?}",
        word_emb.shape(),
        word_emb.dtype()
    );
    assert_eq!(word_emb.shape(), &[50265, 768]);
}
