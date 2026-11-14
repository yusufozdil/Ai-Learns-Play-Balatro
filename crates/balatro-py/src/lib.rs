//! `balatro-py`: PyO3 bindings exposing `balatro-core` to Python as `balatro_rl._engine`.
//!
//! * `VecEnv` (`vec_env.rs`): batched envs, rayon, GIL released, in-place numpy buffers.
//! * `Game` (`single.rs`): one run, for debugging.
//! * `interface.rs`: the TEMPORARY observation/reward/done definitions (Phase 8/10 replace them).

mod interface;
mod single;
mod vec_env;

use pyo3::prelude::*;

/// Version of the underlying rules engine.
#[pyfunction]
fn version() -> &'static str {
    balatro_core::version()
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_class::<vec_env::VecEnv>()?;
    m.add_class::<single::PyGame>()?;
    m.add("OBS_DIM", interface::OBS_DIM)?;
    m.add("DONE_RUNNING", interface::DONE_NO)?;
    m.add("DONE_LOSS", interface::DONE_LOSS)?;
    m.add("DONE_WIN", interface::DONE_WIN)?;
    m.add("DONE_TRUNCATED", interface::DONE_TRUNCATED)?;
    Ok(())
}
