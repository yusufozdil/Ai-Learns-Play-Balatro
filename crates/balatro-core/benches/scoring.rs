//! Criterion benchmarks for `scoring` (filled in by its Phase 1 task).

use criterion::{Criterion, criterion_group, criterion_main};

fn placeholder(_c: &mut Criterion) {}

criterion_group!(benches, placeholder);
criterion_main!(benches);
