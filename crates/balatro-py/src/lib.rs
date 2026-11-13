//! `balatro-py`: PyO3 bindings exposing `balatro-core` to Python as `balatro_rl._engine`.
//! Later phases add `Game` (single env, debugging), `VecEnv` (batched, rayon, GIL released) and
//! observation/action encoding that writes straight into preallocated numpy buffers (Phase 8).

use pyo3::prelude::*;

/// Version of the underlying rules engine.
#[pyfunction]
fn version() -> &'static str {
    balatro_core::version()
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
