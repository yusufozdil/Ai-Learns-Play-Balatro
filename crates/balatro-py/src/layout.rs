//! Buffer layout of the RL interface (see `docs/design/rl_interface.md`, the document of record).
//!
//! * Capacities and vocabularies as compile-time constants.
//! * [`buffers!`]: one table of every caller-owned array (name, dtype, per-env shape) that
//!   generates the raw row accessor [`Bufs`], the pyo3 extraction [`PyBufs`], the Rust-test
//!   storage [`Owned`] and the [`SPECS`] list handed to Python.
//! * Feature name lists, in column order, for `layout()`.
//!
//! Bump [`INTERFACE_VERSION`] on any change of names, shapes, dtypes, feature order, scaling,
//! vocabulary, action or reward semantics.

use balatro_core::content::ids::{BossId, ConsumableId, TagId, VoucherId};
use balatro_core::content::jokers::JokerId;
use balatro_core::data::run_rules::MAX_SELECTABLE_HAND;
use balatro_core::enums::{Edition, Enhancement, HandType, Rank, Seal, Suit};
use numpy::{PyReadwriteArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::marker::PhantomData;

/// Version of the whole interface (buffers, semantics, reward).
pub const INTERFACE_VERSION: u32 = 1;

/// Number of action kinds.
pub const KINDS: usize = 17;
/// Hand tokens (= the core's selectable hand positions).
pub const HAND_CAP: usize = 32;
/// Joker tokens.
pub const JOKER_CAP: usize = 20;
/// Consumable tokens.
pub const CONS_CAP: usize = 8;
/// Shop card slots in the offer array.
pub const SHOP_CARD_CAP: usize = 8;
/// Shop booster slots.
pub const SHOP_PACK_CAP: usize = 4;
/// Shop voucher slots.
pub const SHOP_VOUCHER_CAP: usize = 4;
/// Pack item slots.
pub const PACK_ITEM_CAP: usize = 5;
/// First offer row of the booster slots.
pub const OFFER_PACK0: usize = SHOP_CARD_CAP;
/// First offer row of the voucher slots.
pub const OFFER_VOUCHER0: usize = OFFER_PACK0 + SHOP_PACK_CAP;
/// First offer row of the pack items.
pub const OFFER_ITEM0: usize = OFFER_VOUCHER0 + SHOP_VOUCHER_CAP;
/// Offer tokens.
pub const OFFER_CAP: usize = OFFER_ITEM0 + PACK_ITEM_CAP;
/// Width of `arg0_mask` (the largest addressable source).
pub const MAX_ARG: usize = JOKER_CAP;
/// Card-selection rows: Play, Discard, UseConsumable per slot, PickPackItem per item.
pub const NSEL: usize = 2 + CONS_CAP + PACK_ITEM_CAP;
/// Most cards of a Play/Discard selection (sequential selection steps).
pub const MAX_PLAY_CARDS: usize = 5;

const _: () = assert!(HAND_CAP == MAX_SELECTABLE_HAND);
const _: () = assert!(MAX_ARG >= CONS_CAP && MAX_ARG >= SHOP_CARD_CAP);

/// Scalar columns of `global_f` (names in order); vectors follow, see [`global_f_names`].
pub const GLOBAL_SCALARS: [&str; 27] = [
    "ante",
    "round",
    "target_log",
    "score_log",
    "progress",
    "score_ratio",
    "money",
    "hands_left",
    "discards_left",
    "hand_size",
    "hand_count",
    "joker_count",
    "joker_capacity",
    "consumable_count",
    "consumable_capacity",
    "reroll_price",
    "interest_cap",
    "draw_pile",
    "deck_size",
    "hands_played_round",
    "discards_used_round",
    "boss_active",
    "boss_disabled",
    "jokers_hidden",
    "showdown",
    "won",
    "picks_left",
];

/// Vector blocks of `global_f` after the scalars: `(prefix, length)`.
pub const GLOBAL_VECTORS: [(&str, usize); 11] = [
    ("hand_level", HandType::COUNT),
    ("hand_plays_round", HandType::COUNT),
    ("hand_plays_run", HandType::COUNT),
    ("draw_rank_suit", Rank::COUNT * Suit::COUNT),
    ("draw_stone", 1),
    ("draw_enh", Enhancement::COUNT - 1),
    ("draw_edition", Edition::COUNT - 1),
    ("draw_seal", Seal::COUNT - 1),
    ("voucher", VoucherId::COUNT),
    ("tag", TagId::COUNT),
    ("pending_pack", PACK_SPECS),
];

/// Booster specs (kind x size).
pub const PACK_SPECS: usize = 15;

/// Columns of `global_f`.
pub const GLOBAL_F: usize = GLOBAL_SCALARS.len() + global_vector_len();

const fn global_vector_len() -> usize {
    let mut n = 0;
    let mut i = 0;
    while i < GLOBAL_VECTORS.len() {
        n += GLOBAL_VECTORS[i].1;
        i += 1;
    }
    n
}

/// Names of every `global_f` column.
pub fn global_f_names() -> Vec<String> {
    let mut v: Vec<String> = GLOBAL_SCALARS.iter().map(|s| (*s).to_string()).collect();
    for (p, n) in GLOBAL_VECTORS {
        if n == 1 {
            v.push(p.to_string());
        } else {
            v.extend((0..n).map(|i| format!("{p}.{i}")));
        }
    }
    v
}

/// `global_i` columns with their vocabulary sizes.
pub const GLOBAL_I: [(&str, usize); 8] = [
    ("phase", 6),
    ("blind", 3),
    ("boss", 1 + BossId::COUNT),
    ("last_hand", 1 + HandType::COUNT),
    ("boss_hand", 1 + HandType::COUNT),
    ("open_pack", 1 + PACK_SPECS),
    ("offered_tag", 1 + TagId::COUNT),
    ("offered_tag_hand", 1 + HandType::COUNT),
];

/// Shared item vocabulary: 0 none, jokers, consumables, vouchers, booster specs.
pub const ITEM_VOCAB: usize =
    1 + JokerId::COUNT + ConsumableId::COUNT + VoucherId::COUNT + PACK_SPECS;
/// Item id of a joker.
pub const fn joker_item(i: usize) -> i32 {
    1 + i as i32
}
/// Item id of a consumable.
pub const fn cons_item(i: usize) -> i32 {
    (1 + JokerId::COUNT + i) as i32
}
/// Item id of a voucher.
pub const fn voucher_item(i: usize) -> i32 {
    (1 + JokerId::COUNT + ConsumableId::COUNT + i) as i32
}
/// Item id of a booster spec.
pub const fn pack_item(i: usize) -> i32 {
    (1 + JokerId::COUNT + ConsumableId::COUNT + VoucherId::COUNT + i) as i32
}

/// Rank / suit / enhancement / edition / seal vocabularies of card features.
pub const RANK_VOCAB: usize = Rank::COUNT + 2;
/// See [`RANK_VOCAB`].
pub const SUIT_VOCAB: usize = Suit::COUNT + 2;
/// See [`RANK_VOCAB`].
pub const ENH_VOCAB: usize = Enhancement::COUNT + 1;
/// See [`RANK_VOCAB`].
pub const EDITION_VOCAB: usize = Edition::COUNT + 1;
/// See [`RANK_VOCAB`].
pub const SEAL_VOCAB: usize = Seal::COUNT + 1;

/// `hand_f` columns.
pub const HAND_F: [&str; 6] = [
    "pos",
    "chips",
    "perma_bonus",
    "debuffed",
    "face_down",
    "forced",
];
/// `hand_i` columns with vocabulary sizes.
pub const HAND_I: [(&str, usize); 5] = [
    ("rank", RANK_VOCAB),
    ("suit", SUIT_VOCAB),
    ("enhancement", ENH_VOCAB),
    ("edition", EDITION_VOCAB),
    ("seal", SEAL_VOCAB),
];
/// `joker_f` columns.
pub const JOKER_F: [&str; 13] = [
    "pos",
    "sell_value",
    "eternal",
    "perishable",
    "perish_rounds_left",
    "rental",
    "debuffed",
    "hidden",
    "value_lin",
    "value_log",
    "counter_lin",
    "counter_log",
    "aux_byte",
];
/// `joker_i` columns with vocabulary sizes.
pub const JOKER_I: [(&str, usize); 5] = [
    ("item", ITEM_VOCAB),
    ("edition", EDITION_VOCAB),
    ("aux_suit", 1 + Suit::COUNT),
    ("aux_rank", 1 + Rank::COUNT),
    ("aux_hand", 1 + HandType::COUNT),
];
/// `cons_f` columns.
pub const CONS_F: [&str; 3] = ["pos", "sell_value", "negative"];
/// `cons_i` columns with vocabulary sizes.
pub const CONS_I: [(&str, usize); 1] = [("item", ITEM_VOCAB)];
/// `offer_f` columns.
pub const OFFER_F: [&str; 7] = [
    "price",
    "free",
    "affordable",
    "eternal",
    "perishable",
    "rental",
    "sell_value",
];
/// `offer_i` columns with vocabulary sizes.
pub const OFFER_I: [(&str, usize); 7] = [
    ("source", 5),
    ("item", ITEM_VOCAB),
    ("edition", EDITION_VOCAB),
    ("rank", RANK_VOCAB),
    ("suit", SUIT_VOCAB),
    ("enhancement", ENH_VOCAB),
    ("seal", SEAL_VOCAB),
];

/// dtype name (numpy spelling) of a buffer element type.
pub trait Dt {
    /// numpy dtype name.
    const NAME: &'static str;
}
impl Dt for f32 {
    const NAME: &'static str = "float32";
}
impl Dt for i32 {
    const NAME: &'static str = "int32";
}
impl Dt for u8 {
    const NAME: &'static str = "uint8";
}
impl Dt for u32 {
    const NAME: &'static str = "uint32";
}
impl Dt for bool {
    const NAME: &'static str = "bool";
}

/// One buffer of the interface.
#[derive(Debug, Clone, Copy)]
pub struct BufSpec {
    /// Dict key.
    pub name: &'static str,
    /// numpy dtype name.
    pub dtype: &'static str,
    /// Per-env shape (the batch dimension comes first in the caller's array).
    pub shape: &'static [usize],
}

macro_rules! buffers {
    ($($name:ident : $ty:ty = [$($dim:expr),*]),* $(,)?) => {
        /// Per-env element count of each buffer.
        #[allow(non_upper_case_globals, missing_docs)]
        pub mod stride {
            use super::*;
            $(pub const $name: usize = 1 $(* $dim)*;)*
        }

        /// Every buffer of the interface, in declaration order.
        pub const SPECS: &[BufSpec] = &[
            $(BufSpec { name: stringify!($name), dtype: <$ty as Dt>::NAME, shape: &[$($dim),*] }),*
        ];

        /// Raw handle to caller-owned storage for `n` envs. `Send + Sync` because every env index
        /// is touched by exactly one thread (see [`Bufs::row`]).
        #[allow(missing_docs)]
        pub struct Bufs<'a> {
            n: usize,
            $($name: *mut $ty,)*
            _life: PhantomData<&'a mut ()>,
        }
        // SAFETY: rows of distinct env indices never overlap; callers uphold the contract of `row`.
        unsafe impl Send for Bufs<'_> {}
        unsafe impl Sync for Bufs<'_> {}

        /// The slices of one env.
        #[allow(missing_docs)]
        pub struct Row<'a> {
            $(pub $name: &'a mut [$ty],)*
        }

        impl Bufs<'_> {
            /// Mutable view of env `i`'s slices.
            ///
            /// # Safety
            /// No other live `Row` of the same index may exist (rows of different indices are
            /// disjoint). The parallel loops hand each env index to exactly one thread.
            pub unsafe fn row(&self, i: usize) -> Row<'_> {
                assert!(i < self.n, "env row out of range");
                // SAFETY: in bounds by the assert; disjointness by the caller's contract.
                unsafe {
                    Row {
                        $($name: std::slice::from_raw_parts_mut(self.$name.add(i * stride::$name), stride::$name),)*
                    }
                }
            }
        }

        /// Rust-owned storage with the same layout (tests, the single-run debug class).
        #[allow(missing_docs)]
        pub struct Owned {
            pub n: usize,
            $(pub $name: Vec<$ty>,)*
        }

        impl Owned {
            /// Zeroed storage for `n` envs.
            pub fn new(n: usize) -> Self {
                Owned { n, $($name: vec![<$ty>::default(); n * stride::$name],)* }
            }

            /// Raw handle (borrows `self` mutably for its lifetime).
            pub fn bufs(&mut self) -> Bufs<'_> {
                Bufs { n: self.n, $($name: self.$name.as_mut_ptr(),)* _life: PhantomData }
            }

            /// Copy of the arrays named in `names` as numpy arrays of shape `(n, ...)`, or without
            /// the batch dimension when `squeeze` (`n == 1`).
            pub fn to_py<'py>(&self, py: Python<'py>, names: &[&str], squeeze: bool)
                -> PyResult<Bound<'py, PyDict>>
            {
                use numpy::{PyArray1, PyArrayMethods};
                let d = PyDict::new(py);
                $(
                    if names.contains(&stringify!($name)) {
                        let mut shape: Vec<usize> = Vec::new();
                        if !squeeze { shape.push(self.n); }
                        shape.extend_from_slice(&[$($dim),*]);
                        let a = PyArray1::from_slice(py, &self.$name).reshape(shape)?;
                        d.set_item(stringify!($name), a)?;
                    }
                )*
                Ok(d)
            }
        }

        /// pyo3 side: the extracted, borrow-checked numpy arrays of a call.
        #[allow(missing_docs)]
        pub struct PyBufs<'py> {
            n: usize,
            $($name: PyReadwriteArrayDyn<'py, $ty>,)*
        }

        impl<'py> PyBufs<'py> {
            /// Extract every buffer from `d`, checking dtype, `(n, ...)` shape, writability and
            /// C-contiguity.
            pub fn extract(d: &Bound<'py, PyDict>, n: usize) -> PyResult<Self> {
                Ok(PyBufs {
                    n,
                    $($name: {
                        let item = d.get_item(stringify!($name))?.ok_or_else(|| {
                            PyKeyError::new_err(format!("missing buffer `{}`", stringify!($name)))
                        })?;
                        let arr: PyReadwriteArrayDyn<'py, $ty> = item.extract().map_err(|e| {
                            PyValueError::new_err(format!(
                                "buffer `{}` must be a writable numpy array of dtype {}: {e}",
                                stringify!($name), <$ty as Dt>::NAME
                            ))
                        })?;
                        let want: &[usize] = &[n, $($dim),*];
                        if arr.shape() != want {
                            return Err(PyValueError::new_err(format!(
                                "buffer `{}` has shape {:?}, expected {:?}",
                                stringify!($name), arr.shape(), want
                            )));
                        }
                        // `as_slice_mut` also accepts Fortran order, which would scramble rows.
                        if !arr.is_c_contiguous() {
                            return Err(PyValueError::new_err(format!(
                                "buffer `{}` must be C-contiguous", stringify!($name)
                            )));
                        }
                        arr
                    },)*
                })
            }

            /// Raw handle over the extracted arrays.
            pub fn bufs(&mut self) -> PyResult<Bufs<'_>> {
                Ok(Bufs {
                    n: self.n,
                    $($name: self.$name.as_slice_mut().map_err(|e| {
                        PyValueError::new_err(format!("buffer `{}`: {e}", stringify!($name)))
                    })?.as_mut_ptr(),)*
                    _life: PhantomData,
                })
            }
        }
    };
}

buffers! {
    // --- observation -------------------------------------------------------------------------
    global_f: f32 = [GLOBAL_F],
    global_i: i32 = [GLOBAL_I.len()],
    hand_f: f32 = [HAND_CAP, HAND_F.len()],
    hand_i: i32 = [HAND_CAP, HAND_I.len()],
    hand_valid: bool = [HAND_CAP],
    joker_f: f32 = [JOKER_CAP, JOKER_F.len()],
    joker_i: i32 = [JOKER_CAP, JOKER_I.len()],
    joker_valid: bool = [JOKER_CAP],
    cons_f: f32 = [CONS_CAP, CONS_F.len()],
    cons_i: i32 = [CONS_CAP, CONS_I.len()],
    cons_valid: bool = [CONS_CAP],
    offer_f: f32 = [OFFER_CAP, OFFER_F.len()],
    offer_i: i32 = [OFFER_CAP, OFFER_I.len()],
    offer_valid: bool = [OFFER_CAP],
    // --- masks -------------------------------------------------------------------------------
    kind_mask: bool = [KINDS],
    arg0_mask: bool = [KINDS, MAX_ARG],
    arg1_mask: bool = [MAX_ARG],
    sel_candidates: u32 = [NSEL],
    sel_required: u32 = [NSEL],
    sel_min: u8 = [NSEL],
    sel_max: u8 = [NSEL],
    // --- step outputs ------------------------------------------------------------------------
    reward: f32 = [],
    done: u8 = [],
    invalid: u8 = [],
}

/// Names of the observation buffers.
pub const OBS_NAMES: [&str; 14] = [
    "global_f",
    "global_i",
    "hand_f",
    "hand_i",
    "hand_valid",
    "joker_f",
    "joker_i",
    "joker_valid",
    "cons_f",
    "cons_i",
    "cons_valid",
    "offer_f",
    "offer_i",
    "offer_valid",
];
/// Names of the mask buffers.
pub const MASK_NAMES: [&str; 7] = [
    "kind_mask",
    "arg0_mask",
    "arg1_mask",
    "sel_candidates",
    "sel_required",
    "sel_min",
    "sel_max",
];
/// Names of the per-step output buffers.
pub const OUT_NAMES: [&str; 3] = ["reward", "done", "invalid"];

/// Where `arg0` of each kind points: `(token array, row offset)`; `None` = no `arg0`.
pub const ARG0_POINTER: [Option<(&str, usize)>; KINDS] = [
    None,                            // SelectBlind
    None,                            // SkipBlind
    None,                            // Play
    None,                            // Discard
    Some(("joker", 0)),              // SellJoker
    None,                            // EndShop
    Some(("cons", 0)),               // UseConsumable
    Some(("cons", 0)),               // SellConsumable
    Some(("joker", 0)),              // MoveJoker
    Some(("offer", 0)),              // Buy
    Some(("offer", 0)),              // BuyAndUse
    Some(("offer", OFFER_PACK0)),    // OpenPack
    Some(("offer", OFFER_VOUCHER0)), // BuyVoucher
    None,                            // Reroll
    Some(("offer", OFFER_ITEM0)),    // PickPackItem
    None,                            // SkipPack
    None,                            // RerollBoss
];

/// Source capacity of the `arg0` of each kind (slots at or above it are not addressable);
/// kinds without an `arg0` accept only `{0}`.
pub const fn arg0_cap(kind: usize) -> usize {
    match kind {
        4 | 8 => JOKER_CAP,
        6 | 7 => CONS_CAP,
        9 | 10 => SHOP_CARD_CAP,
        11 => SHOP_PACK_CAP,
        12 => SHOP_VOUCHER_CAP,
        14 => PACK_ITEM_CAP,
        _ => 1,
    }
}

/// Row of the card-selection table of `(kind, arg0)` (`None` if the kind never takes cards or
/// `arg0` is outside the table).
pub fn sel_row(kind: usize, arg0: usize) -> Option<usize> {
    match kind {
        2 => Some(0),
        3 => Some(1),
        6 if arg0 < CONS_CAP => Some(2 + arg0),
        14 if arg0 < PACK_ITEM_CAP => Some(2 + CONS_CAP + arg0),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_layout_is_consistent() {
        assert_eq!(global_f_names().len(), GLOBAL_F);
        assert_eq!(GLOBAL_F, 203);
        assert_eq!(OFFER_CAP, 21);
        assert_eq!(stride::arg0_mask, KINDS * MAX_ARG);
        assert_eq!(SPECS.len(), 24);
        assert_eq!(ITEM_VOCAB, 250);
    }

    #[test]
    fn sel_rows_are_dense_and_distinct() {
        let mut seen = [false; NSEL];
        for (k, a) in [(2, 0), (3, 0)]
            .into_iter()
            .chain((0..CONS_CAP).map(|a| (6, a)))
            .chain((0..PACK_ITEM_CAP).map(|a| (14, a)))
        {
            let r = sel_row(k, a).unwrap();
            assert!(!seen[r]);
            seen[r] = true;
        }
        assert!(seen.iter().all(|&b| b));
    }
}
