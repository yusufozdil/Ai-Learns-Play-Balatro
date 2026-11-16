//! `power`: the build-power meter (`balatro_core::agents::power`) as a free function.

use balatro_core::agents::power::{PowerCfg, power};
use balatro_core::serialize::state_from_json;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Build power of a state against this ante's boss: `power(state, k=8, boss_aware=True,
/// play_on=True, salt=0)`. `state` is a state JSON string or an object with `state_json()` (the
/// `Game`). Returns `{phi, std, frac_beat, target, k}`. Deterministic in (state, salt); the state
/// is not modified.
#[pyfunction]
#[pyo3(name = "power", signature = (state, k=8, boss_aware=true, play_on=true, salt=0))]
pub fn power_py<'py>(
    py: Python<'py>,
    state: &Bound<'py, PyAny>,
    k: usize,
    boss_aware: bool,
    play_on: bool,
    salt: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let json: String = match state.extract::<String>() {
        Ok(s) => s,
        Err(_) => state.call_method0("state_json")?.extract()?,
    };
    let st = state_from_json(&json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let cfg = PowerCfg {
        boss_aware,
        play_on,
        salt,
    };
    let r = py.detach(|| power(&st, k, &cfg));
    let d = PyDict::new(py);
    d.set_item("phi", r.phi)?;
    d.set_item("std", r.std)?;
    d.set_item("frac_beat", r.frac_beat)?;
    d.set_item("target", r.target)?;
    d.set_item("k", r.k)?;
    Ok(d)
}
