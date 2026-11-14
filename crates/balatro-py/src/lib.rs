//! `balatro-py`: PyO3 bindings exposing `balatro-core` to Python as `balatro_rl._engine`.
//!
//! The RL interface is specified in `docs/design/rl_interface.md`:
//! * `layout.rs`: capacities, vocabularies, the table of caller-owned buffers.
//! * `encode.rs`: token observation and mask writers.
//! * `env.rs`: one env (structured-action step, reward, auto-reset, samplers, teacher).
//! * `vec_env.rs`: `VecEnv`, batched envs, rayon, GIL released.
//! * `single.rs`: `Game`, one run for debugging and the tests.

mod encode;
mod env;
mod layout;
mod single;
mod vec_env;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

/// Version of the underlying rules engine.
#[pyfunction]
fn version() -> &'static str {
    balatro_core::version()
}

fn str_list<'py>(
    py: Python<'py>,
    it: impl IntoIterator<Item = String>,
) -> PyResult<Bound<'py, PyList>> {
    PyList::new(py, it.into_iter().collect::<Vec<_>>())
}

fn named<'py>(
    py: Python<'py>,
    cols: &[(&str, usize)],
) -> PyResult<(Bound<'py, PyList>, Vec<usize>)> {
    Ok((
        str_list(py, cols.iter().map(|c| c.0.to_string()))?,
        cols.iter().map(|c| c.1).collect(),
    ))
}

/// The interface description: capacities, buffer specs, feature names, vocabulary sizes,
/// pointer targets (everything the model needs to size itself).
#[pyfunction]
#[pyo3(name = "layout")]
fn layout_py(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    use layout::*;
    let d = PyDict::new(py);
    d.set_item("version", INTERFACE_VERSION)?;
    let caps = PyDict::new(py);
    for (k, v) in [
        ("kinds", KINDS),
        ("hand", HAND_CAP),
        ("joker", JOKER_CAP),
        ("cons", CONS_CAP),
        ("offer", OFFER_CAP),
        ("shop_cards", SHOP_CARD_CAP),
        ("shop_packs", SHOP_PACK_CAP),
        ("shop_vouchers", SHOP_VOUCHER_CAP),
        ("pack_items", PACK_ITEM_CAP),
        ("offer_pack0", OFFER_PACK0),
        ("offer_voucher0", OFFER_VOUCHER0),
        ("offer_item0", OFFER_ITEM0),
        ("max_arg", MAX_ARG),
        ("nsel", NSEL),
        ("max_play_cards", MAX_PLAY_CARDS),
        ("item_vocab", ITEM_VOCAB),
    ] {
        caps.set_item(k, v)?;
    }
    d.set_item("caps", caps)?;
    let bufs = PyList::empty(py);
    for s in SPECS {
        let shape = PyTuple::new(py, s.shape.iter().copied())?;
        bufs.append((s.name, s.dtype, shape))?;
    }
    d.set_item("buffers", bufs)?;
    d.set_item(
        "obs_names",
        str_list(py, OBS_NAMES.iter().map(|s| s.to_string()))?,
    )?;
    d.set_item(
        "mask_names",
        str_list(py, MASK_NAMES.iter().map(|s| s.to_string()))?,
    )?;
    d.set_item(
        "out_names",
        str_list(py, OUT_NAMES.iter().map(|s| s.to_string()))?,
    )?;
    // Feature names and vocabulary sizes.
    let names = PyDict::new(py);
    let vocab = PyDict::new(py);
    names.set_item("global_f", str_list(py, global_f_names())?)?;
    let (n, v) = named(py, &GLOBAL_I)?;
    names.set_item("global_i", n)?;
    vocab.set_item("global_i", v)?;
    names.set_item(
        "hand_f",
        str_list(py, HAND_F.iter().map(|s| s.to_string()))?,
    )?;
    let (n, v) = named(py, &HAND_I)?;
    names.set_item("hand_i", n)?;
    vocab.set_item("hand_i", v)?;
    names.set_item(
        "joker_f",
        str_list(py, JOKER_F.iter().map(|s| s.to_string()))?,
    )?;
    let (n, v) = named(py, &JOKER_I)?;
    names.set_item("joker_i", n)?;
    vocab.set_item("joker_i", v)?;
    names.set_item(
        "cons_f",
        str_list(py, CONS_F.iter().map(|s| s.to_string()))?,
    )?;
    let (n, v) = named(py, &CONS_I)?;
    names.set_item("cons_i", n)?;
    vocab.set_item("cons_i", v)?;
    names.set_item(
        "offer_f",
        str_list(py, OFFER_F.iter().map(|s| s.to_string()))?,
    )?;
    let (n, v) = named(py, &OFFER_I)?;
    names.set_item("offer_i", n)?;
    vocab.set_item("offer_i", v)?;
    d.set_item("feature_names", names)?;
    d.set_item("vocab", vocab)?;
    // Pointer targets of arg0 per kind: (token array, row offset, source capacity) or None.
    let ptr = PyList::empty(py);
    for (k, p) in ARG0_POINTER.iter().enumerate() {
        match p {
            Some((tok, off)) => ptr.append((*tok, *off, arg0_cap(k)))?,
            None => ptr.append(py.None())?,
        }
    }
    d.set_item("arg0_pointer", ptr)?;
    Ok(d)
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(layout_py, m)?)?;
    m.add_class::<vec_env::VecEnv>()?;
    m.add_class::<single::PyGame>()?;
    m.add("INTERFACE_VERSION", layout::INTERFACE_VERSION)?;
    m.add("DONE_RUNNING", env::DONE_NO)?;
    m.add("DONE_LOSS", env::DONE_LOSS)?;
    m.add("DONE_WIN", env::DONE_WIN)?;
    m.add("DONE_TRUNCATED", env::DONE_TRUNCATED)?;
    Ok(())
}
