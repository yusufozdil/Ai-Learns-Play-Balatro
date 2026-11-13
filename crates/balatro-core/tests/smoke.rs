//! Smoke test: crate links and exposes a version.

#[test]
fn version_is_set() {
    assert!(!balatro_core::version().is_empty());
}
