use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, TransactionBehavior};
use thiserror::Error;

use crate::core::types::{Degradation, DegradationKind, Embedding};

/// Chunk size for `IN` queries — safely below SQLite's 999-variable limit.
const CHUNK_SIZE: usize = 900;
/// Current schema version stored in `PRAGMA user_version`.
const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Error)]
pub(crate) enum CacheError {
    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("corrupt cache: {0}")]
    Corrupt(String),
}

/// SQLite-backed embedding cache with little-endian f32 BLOB storage (DD6).
///
/// Schema: single table `embeddings (key TEXT PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL)`.
/// WAL journal mode with `PRAGMA busy_timeout = 5000` for concurrent access (DD12).
/// Self-heals on schema version mismatch or database corruption (DD12).
pub(crate) struct EmbeddingCache {
    /// `RefCell` provides interior mutability: `get_many`/`set_many` take `&self`
    /// (matching the plan API and Python's duck-typed interface) while rusqlite
    /// requires `&mut Connection` for transactions.
    conn: RefCell<Connection>,
    root: PathBuf,
    degradations: Vec<Degradation>,
}

impl EmbeddingCache {
    /// Open or create the cache database. Expands `~` prefix. Self-heals on corruption/version mismatch.
    pub(crate) fn new(root: &str) -> Result<Self, CacheError> {
        let root_path = expand_tilde(root)?;
        fs::create_dir_all(&root_path)?;
        let db_path = root_path.join("embeddings.sqlite3");
        let mut degradations = Vec::new();
        let conn = open_with_self_heal(&db_path, &mut degradations)?;
        Ok(Self {
            conn: RefCell::new(conn),
            root: root_path,
            degradations,
        })
    }

    /// Batch lookup. Returns only found entries. Chunked to respect SQLite variable limits.
    /// Lazily migrates legacy JSON files for cache misses.
    pub(crate) fn get_many(&self, keys: &[&str]) -> Result<HashMap<String, Embedding>, CacheError> {
        if keys.is_empty() {
            return Ok(HashMap::new());
        }

        let mut results: HashMap<String, Embedding> = HashMap::new();

        for chunk in keys.chunks(CHUNK_SIZE) {
            let placeholders = chunk.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = format!("SELECT key, dim, vec FROM embeddings WHERE key IN ({placeholders})");
            let params = rusqlite::params_from_iter(chunk.iter().copied());
            let conn = self.conn.borrow();
            let mut stmt = conn.prepare(&sql)?;
            let rows = stmt.query_map(params, |row| {
                let key: String = row.get(0)?;
                let dim: usize = row.get::<_, i64>(1)? as usize;
                let blob: Vec<u8> = row.get(2)?;
                Ok((key, dim, blob))
            })?;
            for row in rows {
                let (key, dim, blob) = row?;
                let vector = blob_to_vec(&blob);
                // Guard against corrupt BLOBs: a truncated blob produces fewer floats
                // than the stored `dim` column, which would cause dimension-mismatch panics.
                debug_assert!(
                    vector.len() == dim,
                    "BLOB length mismatch: got {} floats, expected dim={}",
                    vector.len(),
                    dim
                );
                results.insert(key, Embedding { vector });
            }
        }

        // Lazy migration: check legacy JSON files for cache misses
        let missed: Vec<&str> = keys
            .iter()
            .copied()
            .filter(|k| !results.contains_key(*k))
            .collect();
        if !missed.is_empty() {
            let migrated = self.migrate_json(&missed)?;
            results.extend(migrated);
        }

        Ok(results)
    }

    /// Batch upsert. Serializes vectors as little-endian f32 BLOBs (DD6).
    pub(crate) fn set_many(&self, items: &HashMap<String, Embedding>) -> Result<(), CacheError> {
        if items.is_empty() {
            return Ok(());
        }

        // Use an immediate transaction so concurrent writers get SQLITE_BUSY (not SQLITE_LOCKED),
        // giving busy_timeout a chance to retry rather than failing immediately.
        let mut conn = self.conn.borrow_mut();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO embeddings(key, dim, vec) VALUES(?,?,?) \
                 ON CONFLICT(key) DO UPDATE SET dim=excluded.dim, vec=excluded.vec",
            )?;
            for (key, emb) in items {
                let blob = vec_to_blob(&emb.vector);
                stmt.execute(rusqlite::params![key, emb.vector.len() as i64, blob])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Drain accumulated degradation events (cache self-heal, etc.).
    pub(crate) fn take_degradations(&mut self) -> Vec<Degradation> {
        std::mem::take(&mut self.degradations)
    }

    /// Lazily migrate legacy `{safe_key}.json` files into SQLite.
    fn migrate_json(&self, keys: &[&str]) -> Result<HashMap<String, Embedding>, CacheError> {
        let mut found: HashMap<String, Embedding> = HashMap::new();
        for &key in keys {
            let safe_key = key.replace('/', "_");
            let json_path = self.root.join(format!("{safe_key}.json"));
            if !json_path.exists() {
                continue;
            }
            if let Ok(content) = fs::read_to_string(&json_path) {
                if let Ok(payload) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let (Some(vec_val), Some(dim_val)) =
                        (payload.get("vector"), payload.get("dim"))
                    {
                        if let (Some(arr), Some(dim)) = (vec_val.as_array(), dim_val.as_u64()) {
                            let vector: Vec<f32> = arr
                                .iter()
                                .filter_map(|v| v.as_f64().map(|f| f as f32))
                                .collect();
                            // Skip entries where null/non-numeric JSON elements reduced
                            // the vector below the stored dim (invariant: vector.len() == dim).
                            if vector.len() != dim as usize {
                                continue;
                            }
                            found.insert(key.to_owned(), Embedding { vector });
                        }
                    }
                }
            }
        }
        if !found.is_empty() {
            self.set_many(&found)?;
        }
        Ok(found)
    }
}

// ── Serialization helpers ──────────────────────────────────────────────────────

/// Serialize a `Vec<f32>` to little-endian bytes (DD6).
fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Deserialize little-endian bytes to `Vec<f32>` (DD6).
fn blob_to_vec(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

// ── Path helpers ───────────────────────────────────────────────────────────────

/// Expand a leading `~` to the user's home directory.
fn expand_tilde(path: &str) -> Result<PathBuf, CacheError> {
    expand_tilde_with_home(path, dirs::home_dir())
}

/// Current Unix timestamp (seconds) for backup file names.
fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ── Database lifecycle ─────────────────────────────────────────────────────────

/// Open the SQLite database, applying self-heal logic for corrupt/version-mismatched files.
fn open_with_self_heal(
    db_path: &Path,
    degradations: &mut Vec<Degradation>,
) -> Result<Connection, CacheError> {
    match try_open(db_path) {
        Ok(conn) => Ok(conn),
        Err(SelfHealAction::VersionMismatch(version)) => {
            let ts = unix_now();
            let backup = db_path.with_extension(format!("v{version}-{ts}.bak"));
            rename_with_wal(db_path, &backup);
            degradations.push(Degradation {
                kind: DegradationKind::CacheSelfHeal,
                message: format!(
                    "cache schema version {version} != {SCHEMA_VERSION}; backed up to {}",
                    backup.display()
                ),
            });
            // Recurse once — fresh DB will have version=0 and be initialized cleanly
            try_open(db_path).map_err(|e| match e {
                SelfHealAction::DatabaseError(e) => CacheError::Database(e),
                _ => CacheError::Corrupt("self-heal failed on fresh database".into()),
            })
        }
        Err(SelfHealAction::DatabaseError(e)) => {
            let ts = unix_now();
            let backup = db_path.with_extension(format!("corrupt-{ts}.bak"));
            rename_with_wal(db_path, &backup);
            degradations.push(Degradation {
                kind: DegradationKind::CacheSelfHeal,
                message: format!(
                    "corrupt cache backed up to {}; error: {e}",
                    backup.display()
                ),
            });
            try_open(db_path).map_err(|e2| match e2 {
                SelfHealAction::DatabaseError(e2) => CacheError::Database(e2),
                _ => CacheError::Corrupt("self-heal failed on fresh database".into()),
            })
        }
    }
}

enum SelfHealAction {
    VersionMismatch(u32),
    DatabaseError(rusqlite::Error),
}

/// Try to open the database and initialize/validate the schema.
fn try_open(db_path: &Path) -> Result<Connection, SelfHealAction> {
    let conn = Connection::open(db_path).map_err(SelfHealAction::DatabaseError)?;
    apply_pragmas(&conn).map_err(SelfHealAction::DatabaseError)?;

    let version: u32 = conn
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(SelfHealAction::DatabaseError)?;

    if version == 0 {
        // Fresh database — create schema
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS embeddings \
             (key TEXT PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL);",
        )
        .map_err(SelfHealAction::DatabaseError)?;
        conn.execute_batch(&format!("PRAGMA user_version = {SCHEMA_VERSION};"))
            .map_err(SelfHealAction::DatabaseError)?;
    } else if version != SCHEMA_VERSION {
        return Err(SelfHealAction::VersionMismatch(version));
    }

    Ok(conn)
}

fn apply_pragmas(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        "PRAGMA journal_mode=WAL; \
         PRAGMA synchronous=NORMAL; \
         PRAGMA temp_store=MEMORY; \
         PRAGMA busy_timeout=5000;",
    )
}

/// Rename `db_path` to `backup`, also renaming WAL and SHM auxiliary files if present.
/// Silently ignores errors (concurrent process may have already renamed).
fn rename_with_wal(db_path: &Path, backup: &Path) {
    let _ = fs::rename(db_path, backup);
    // WAL auxiliary files share the same base name
    for suffix in ["-wal", "-shm"] {
        let aux_src = PathBuf::from(format!("{}{suffix}", db_path.display()));
        if aux_src.exists() {
            let aux_dst = PathBuf::from(format!("{}{suffix}", backup.display()));
            let _ = fs::rename(aux_src, aux_dst);
        }
    }
}

// ── tilde expansion core (parameterized on home dir for testability) ──────────

/// Expand a `~`-prefixed path given an explicit home dir (None → error for `~` paths).
/// `expand_tilde` calls this with `dirs::home_dir()`; tests pass a fixed home / None.
fn expand_tilde_with_home(path: &str, home: Option<PathBuf>) -> Result<PathBuf, CacheError> {
    if let Some(rest) = path.strip_prefix("~/") {
        let home = home.ok_or_else(|| {
            CacheError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("cannot resolve home directory for cache path '{path}'"),
            ))
        })?;
        Ok(home.join(rest))
    } else if path == "~" {
        home.ok_or_else(|| {
            CacheError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("cannot resolve home directory for cache path '{path}'"),
            ))
        })
    } else {
        Ok(PathBuf::from(path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::make_embedding;
    use tempfile::TempDir;

    fn cache_in(dir: &TempDir) -> EmbeddingCache {
        EmbeddingCache::new(dir.path().to_str().unwrap()).unwrap()
    }

    #[test]
    fn cache_roundtrip() {
        let dir = TempDir::new().unwrap();
        let cache = cache_in(&dir);
        let emb = make_embedding(vec![0.1f32, 0.2, 0.3]);
        let mut items = HashMap::new();
        items.insert("key1".to_string(), emb.clone());
        cache.set_many(&items).unwrap();

        let result = cache.get_many(&["key1"]).unwrap();
        assert_eq!(result.len(), 1);
        // f32 → LE bytes → f32 is lossless for finite values (DD6)
        assert_eq!(result["key1"].vector, emb.vector);
    }

    #[test]
    fn cache_overwrite() {
        let dir = TempDir::new().unwrap();
        let cache = cache_in(&dir);
        let mut items = HashMap::new();
        items.insert("k".to_string(), make_embedding(vec![1.0, 2.0]));
        cache.set_many(&items).unwrap();
        let mut items2 = HashMap::new();
        items2.insert("k".to_string(), make_embedding(vec![9.0, 8.0]));
        cache.set_many(&items2).unwrap();

        let result = cache.get_many(&["k"]).unwrap();
        assert_eq!(result["k"].vector, vec![9.0f32, 8.0]);
    }

    #[test]
    fn cache_missing_keys() {
        let dir = TempDir::new().unwrap();
        let cache = cache_in(&dir);
        let mut items = HashMap::new();
        items.insert("present".to_string(), make_embedding(vec![0.5]));
        cache.set_many(&items).unwrap();

        let result = cache.get_many(&["present", "absent"]).unwrap();
        assert!(result.contains_key("present"));
        assert!(!result.contains_key("absent"));
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn cache_empty_keys() {
        let dir = TempDir::new().unwrap();
        let cache = cache_in(&dir);
        let result = cache.get_many(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn cache_batch_write_read() {
        let dir = TempDir::new().unwrap();
        let cache = cache_in(&dir);
        let mut items = HashMap::new();
        for i in 0..50usize {
            items.insert(
                format!("key{i}"),
                make_embedding(vec![i as f32, i as f32 + 1.0]),
            );
        }
        cache.set_many(&items).unwrap();

        let keys: Vec<&str> = items.keys().map(|s| s.as_str()).collect();
        let result = cache.get_many(&keys).unwrap();
        assert_eq!(result.len(), 50);
        for i in 0..50usize {
            let k = format!("key{i}");
            assert_eq!(result[&k].vector, vec![i as f32, i as f32 + 1.0]);
        }
    }

    #[test]
    fn cache_chunked_read() {
        let dir = TempDir::new().unwrap();
        let cache = cache_in(&dir);
        let mut items = HashMap::new();
        for i in 0..1000usize {
            items.insert(format!("key{i:04}"), make_embedding(vec![i as f32]));
        }
        cache.set_many(&items).unwrap();

        let keys: Vec<String> = (0..1000usize).map(|i| format!("key{i:04}")).collect();
        let key_refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
        let result = cache.get_many(&key_refs).unwrap();
        assert_eq!(result.len(), 1000);
    }

    #[test]
    fn cache_json_migration() {
        let dir = TempDir::new().unwrap();
        let root = dir.path().to_str().unwrap();

        // Write a legacy JSON cache file
        let key = "sha256_abc123";
        let safe_key = key.replace('/', "_");
        let json_path = dir.path().join(format!("{safe_key}.json"));
        let payload = serde_json::json!({ "vector": [0.1f64, 0.2, 0.3], "dim": 3 });
        std::fs::write(&json_path, serde_json::to_string(&payload).unwrap()).unwrap();

        let cache = EmbeddingCache::new(root).unwrap();
        // First read: should migrate from JSON to SQLite
        let result = cache.get_many(&[key]).unwrap();
        assert!(result.contains_key(key), "legacy JSON should be migrated");
        assert_eq!(result[key].vector.len(), 3);

        // Second read: should come from SQLite (JSON still present, but SQLite has it now)
        let result2 = cache.get_many(&[key]).unwrap();
        assert!(result2.contains_key(key));
        assert_eq!(result2[key].vector.len(), 3);
    }

    #[test]
    fn cache_schema_version_mismatch() {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("embeddings.sqlite3");

        // Pre-create a DB with an incompatible schema version
        {
            let conn = Connection::open(&db_path).unwrap();
            conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA user_version = 999;")
                .unwrap();
        }

        let mut cache = EmbeddingCache::new(dir.path().to_str().unwrap()).unwrap();
        // The old DB should be backed up
        let bak_files: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".bak"))
            .collect();
        assert!(
            !bak_files.is_empty(),
            "backup file should exist after version mismatch"
        );

        // A CacheSelfHeal degradation should be recorded
        let degs = cache.take_degradations();
        assert!(
            degs.iter()
                .any(|d| d.kind == DegradationKind::CacheSelfHeal)
        );

        // The new cache should be functional
        let mut items = HashMap::new();
        items.insert("test".to_string(), make_embedding(vec![1.0]));
        cache.set_many(&items).unwrap();
        let result = cache.get_many(&["test"]).unwrap();
        assert!(result.contains_key("test"));
    }

    #[test]
    fn cache_corruption_self_heal() {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("embeddings.sqlite3");

        // Write garbage bytes to simulate a corrupt DB
        std::fs::write(&db_path, b"not a sqlite database at all").unwrap();

        let mut cache = EmbeddingCache::new(dir.path().to_str().unwrap()).unwrap();
        let bak_files: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".bak"))
            .collect();
        assert!(
            !bak_files.is_empty(),
            "backup file should exist after corruption"
        );

        let degs = cache.take_degradations();
        assert!(
            degs.iter()
                .any(|d| d.kind == DegradationKind::CacheSelfHeal)
        );

        // Cache should now be functional
        let mut items = HashMap::new();
        items.insert("k".to_string(), make_embedding(vec![2.0]));
        cache.set_many(&items).unwrap();
        let result = cache.get_many(&["k"]).unwrap();
        assert!(result.contains_key("k"));
    }

    #[test]
    fn cache_tilde_expansion() {
        // Verify that ~/some/path is expanded to home_dir()/some/path
        let home = dirs::home_dir().expect("home dir must exist for this test");
        let expanded = expand_tilde("~/.cache/clonehunter").unwrap();
        assert!(expanded.starts_with(&home));
        assert!(expanded.ends_with(".cache/clonehunter"));
    }

    #[test]
    fn cache_no_tilde_passthrough() {
        // Paths without ~ are used as-is
        let result = expand_tilde("/tmp/test_clonehunter_cache").unwrap();
        assert_eq!(result, PathBuf::from("/tmp/test_clonehunter_cache"));
    }

    #[test]
    fn expand_tilde_with_home_none_errors() {
        // Test the home-dir-None edge case via the helper that accepts Option<PathBuf>
        let result = expand_tilde_with_home("~/foo", None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("home directory"));
    }

    #[test]
    fn expand_tilde_with_home_some_expands() {
        let home = PathBuf::from("/custom/home");
        let result = expand_tilde_with_home("~/bar/baz", Some(home.clone())).unwrap();
        assert_eq!(result, home.join("bar/baz"));
    }
}
