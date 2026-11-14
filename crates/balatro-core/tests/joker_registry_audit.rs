//! Phase 3.R audit: every joker definition must agree with `docs/mechanics/jokers.csv` (name,
//! rarity, base cost, Blueprint compatibility = CSV `copyable`), and the CSV `status` column
//! must agree with the registry (`implemented` <=> a definition exists). The CSV is the
//! inventory the orchestrator tracks progress with; a mismatch means one of the two is wrong.

use balatro_core::content::jokers::{JokerId, def_count, try_def};
use balatro_core::enums::Rarity;

const CSV: &str = include_str!("../../../docs/mechanics/jokers.csv");

/// Minimal RFC 4180 line parser: fields separated by commas, a field may be double-quoted
/// (then commas are literal and `""` is an escaped quote). Records never span lines here.
fn parse_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut cur = String::new();
    let mut quoted = false;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        match (ch, quoted) {
            ('"', true) if chars.peek() == Some(&'"') => {
                cur.push('"');
                chars.next();
            }
            ('"', true) => quoted = false,
            ('"', false) if cur.is_empty() => quoted = true,
            (',', false) => fields.push(std::mem::take(&mut cur)),
            (c, _) => cur.push(c),
        }
    }
    fields.push(cur);
    fields
}

struct Row {
    n: usize,
    id: String,
    name: String,
    rarity: String,
    cost: u32,
    copyable: String,
    status: String,
}

fn rows() -> Vec<Row> {
    let mut lines = CSV.lines().filter(|l| !l.trim().is_empty());
    let header = parse_line(lines.next().expect("header"));
    let col = |name: &str| {
        header
            .iter()
            .position(|h| h.trim() == name)
            .unwrap_or_else(|| panic!("column {name} missing"))
    };
    let (n, id, name, rarity, cost, copyable, status) = (
        col("n"),
        col("id"),
        col("name"),
        col("rarity"),
        col("cost"),
        col("copyable"),
        col("status"),
    );
    lines
        .map(|l| {
            let f = parse_line(l);
            assert_eq!(f.len(), header.len(), "field count in line: {l}");
            Row {
                n: f[n].trim().parse().expect("n"),
                id: f[id].trim().to_string(),
                name: f[name].trim().to_string(),
                rarity: f[rarity].trim().to_string(),
                cost: f[cost].trim().parse().expect("cost"),
                copyable: f[copyable].trim().to_string(),
                status: f[status].trim().to_string(),
            }
        })
        .collect()
}

fn rarity_name(r: Rarity) -> &'static str {
    match r {
        Rarity::Common => "Common",
        Rarity::Uncommon => "Uncommon",
        Rarity::Rare => "Rare",
        Rarity::Legendary => "Legendary",
    }
}

#[test]
fn csv_parser_handles_quotes() {
    assert_eq!(parse_line("a,\"b, c\",d"), vec!["a", "b, c", "d"]);
    assert_eq!(
        parse_line("\"x \"\"q\"\" y\",,z"),
        vec!["x \"q\" y", "", "z"]
    );
}

#[test]
fn csv_lists_every_joker_id_in_index_order() {
    let rows = rows();
    assert_eq!(rows.len(), JokerId::COUNT);
    for (i, (row, id)) in rows.iter().zip(JokerId::ALL).enumerate() {
        assert_eq!(row.n, i + 1, "row number of {}", row.id);
        assert_eq!(
            row.id,
            format!("{id:?}"),
            "CSV order must match JokerId::index()"
        );
    }
}

#[test]
fn every_definition_matches_the_csv() {
    let mut problems = Vec::new();
    for (row, id) in rows().iter().zip(JokerId::ALL) {
        let defs = def_count(id);
        let implemented = row.status == "implemented";
        match (try_def(id), implemented) {
            (Some(_), false) => problems.push(format!(
                "{}: defined but CSV status is `{}`",
                row.id, row.status
            )),
            (None, true) => problems.push(format!("{}: CSV says implemented, no def", row.id)),
            _ => {}
        }
        if defs > 1 {
            problems.push(format!("{}: {defs} definitions", row.id));
        }
        let Some(d) = try_def(id) else { continue };
        if d.name != row.name {
            problems.push(format!(
                "{}: name {:?} vs CSV {:?}",
                row.id, d.name, row.name
            ));
        }
        if rarity_name(d.rarity) != row.rarity {
            problems.push(format!(
                "{}: rarity {:?} vs CSV {}",
                row.id, d.rarity, row.rarity
            ));
        }
        if d.base_cost != row.cost {
            problems.push(format!(
                "{}: cost {} vs CSV {}",
                row.id, d.base_cost, row.cost
            ));
        }
        let copyable = match row.copyable.as_str() {
            "yes" => true,
            "no" => false,
            other => {
                problems.push(format!("{}: copyable `{other}` is not yes/no", row.id));
                continue;
            }
        };
        if d.blueprint_compatible != copyable {
            problems.push(format!(
                "{}: blueprint_compatible {} vs CSV copyable {}",
                row.id, d.blueprint_compatible, row.copyable
            ));
        }
        // Only the two copiers carry a copy resolver.
        let is_copier = matches!(id, JokerId::Blueprint | JokerId::Brainstorm);
        if d.copies.is_some() != is_copier {
            problems.push(format!("{}: copies = {:?}", row.id, d.copies.is_some()));
        }
    }
    assert!(
        problems.is_empty(),
        "{} mismatches:\n{}",
        problems.len(),
        problems.join("\n")
    );
}

#[test]
fn blueprint_incompatible_set_matches_the_contract() {
    // docs/design/hooks.md §8: exactly these 29 are not copyable.
    use JokerId as J;
    let expected = [
        J::FourFingers,
        J::CreditCard,
        J::ChaosTheClown,
        J::DelayedGratification,
        J::Pareidolia,
        J::Egg,
        J::Splash,
        J::SixthSense,
        J::Shortcut,
        J::Cloud9,
        J::Rocket,
        J::MidasMask,
        J::GiftCard,
        J::TurtleBean,
        J::ToTheMoon,
        J::Juggler,
        J::Drunkard,
        J::GoldenJoker,
        J::TradingCard,
        J::MrBones,
        J::Troubadour,
        J::SmearedJoker,
        J::Showman,
        J::MerryAndy,
        J::OopsAll6s,
        J::InvisibleJoker,
        J::Satellite,
        J::Astronomer,
        J::Chicot,
    ];
    assert_eq!(expected.len(), 29);
    let csv_no: Vec<String> = rows()
        .into_iter()
        .filter(|r| r.copyable == "no")
        .map(|r| r.id)
        .collect();
    let mut want: Vec<String> = expected.iter().map(|id| format!("{id:?}")).collect();
    let mut got = csv_no.clone();
    want.sort();
    got.sort();
    assert_eq!(got, want, "CSV copyable=no set differs from hooks.md §8");
}
