#![allow(clippy::needless_borrows_for_generic_args, unused_variables)]
#![cfg(feature = "mlx")]

#[test]
fn mlx_deep_clone_and_eval() {
    let home = dirs::home_dir().unwrap();
    let path = home.join(".cache/huggingface/hub/models--microsoft--codebert-base/snapshots/3b0952feddeffad0063f274080e3c23d75e7eb39/model.safetensors");
    if !path.exists() {
        eprintln!("Skipping: not found");
        return;
    }

    let weights = mlx_rs::Array::load_safetensors(&path).unwrap();
    eprintln!("Loaded {} weight tensors", weights.len());

    // Test deep_clone on one weight
    let w = weights.get("embeddings.word_embeddings.weight").unwrap();
    eprintln!("Original shape: {:?}", w.shape());
    let cloned = w.deep_clone();
    eprintln!("Cloned shape: {:?}", cloned.shape());
    cloned.eval().unwrap();
    let data: &[f32] = cloned.as_slice();
    eprintln!("First 3 values: {:?}", &data[..3]);

    // Deep clone ALL weights
    eprintln!("Deep cloning all weights...");
    let mut cloned_weights: std::collections::HashMap<String, mlx_rs::Array> =
        std::collections::HashMap::new();
    for (key, arr) in &weights {
        cloned_weights.insert(key.clone(), arr.deep_clone());
    }
    eprintln!("Cloned {} weights", cloned_weights.len());

    // Eval all
    eprintln!("Evaluating all cloned weights...");
    for (key, arr) in &cloned_weights {
        arr.eval().unwrap();
    }
    eprintln!("All weights evaluated successfully");
}
