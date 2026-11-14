//! Text-based debug player (development tool only, not a game UI). Phase 2C.
//!
//! Usage: `play_cli [--seed N]`; commands are read from stdin, one per line (scriptable):
//! `select`, `skip`, `play 0 2 5`, `discard 1 3`, `sell <i>`, `end`, `legal`, `state`,
//! `save <file>`, `load <file>`, `replay-save <file>`, `help`, `quit`.

use balatro_core::actions::{Action, ActionError};
use balatro_core::card::Card;
use balatro_core::enums::{Edition, Enhancement, Phase, Rank, Seal, Suit};
use balatro_core::game::{Game, StepInfo};
use balatro_core::serialize::{Replay, state_from_json, state_to_json};
use balatro_core::state::{GameConfig, GameState};
use std::io::{self, BufRead, Write};

/// Run state of the CLI: the game plus the actions recorded since the start of the run.
struct Session {
    game: Game,
    replay: Replay,
    /// False after `load` (history before the loaded state is unknown, so no replay).
    replay_valid: bool,
}

impl Session {
    fn new(seed: u64) -> Self {
        let config = GameConfig::default();
        Session {
            game: Game::new(seed, config),
            replay: Replay::record(seed, config),
            replay_valid: true,
        }
    }
}

fn rank_str(r: Rank) -> String {
    match r {
        Rank::Ten => "10".into(),
        Rank::Jack => "J".into(),
        Rank::Queen => "Q".into(),
        Rank::King => "K".into(),
        Rank::Ace => "A".into(),
        other => (other as u8).to_string(),
    }
}

fn suit_char(s: Suit) -> char {
    match s {
        Suit::Spades => 's',
        Suit::Hearts => 'h',
        Suit::Clubs => 'c',
        Suit::Diamonds => 'd',
    }
}

fn card_str(c: &Card) -> String {
    let mut s = format!("{}{}", rank_str(c.rank), suit_char(c.suit));
    if c.enhancement != Enhancement::None {
        s.push_str(&format!("[{:?}]", c.enhancement));
    }
    if c.edition != Edition::Base {
        s.push_str(&format!("[{:?}]", c.edition));
    }
    if c.seal != Seal::None {
        s.push_str(&format!("[{:?} seal]", c.seal));
    }
    if c.debuffed {
        s.push_str("[debuffed]");
    }
    if c.face_down {
        s.push_str("[face down]");
    }
    s
}

fn shop_text(s: &GameState) -> String {
    use balatro_core::shop::{self, ShopCardKind};
    let mut out = String::from("shop cards:");
    for (i, c) in s.shop.cards.iter().enumerate() {
        if let Some(c) = c {
            let what = match &c.kind {
                ShopCardKind::Joker(j) => format!("{:?} {:?}", j.id, j.edition),
                ShopCardKind::Consumable(id) => format!("{id:?}"),
                ShopCardKind::PlayingCard(card) => card_str(card),
            };
            out.push_str(&format!(" [{i}] {what} ${}", shop::card_price(s, c)));
        }
    }
    out.push_str("\npacks:");
    for (i, p) in s.shop.packs.iter().enumerate() {
        if let Some(p) = p {
            out.push_str(&format!(
                " [{i}] {:?} {:?} ${}",
                p.spec.kind,
                p.spec.size,
                shop::pack_price_of(s, p)
            ));
        }
    }
    out.push_str("\nvouchers:");
    for (i, v) in s.shop.vouchers.iter().enumerate() {
        if let Some(v) = v {
            out.push_str(&format!(
                " [{i}] {:?} ${}",
                v.id,
                shop::voucher_price(s, v.id)
            ));
        }
    }
    out.push_str(&format!("\nreroll ${}\n", shop::reroll_price(s)));
    out
}

fn summary(s: &GameState) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "ante {} round {} blind {:?} | phase {:?}",
        s.ante, s.round, s.blind, s.phase
    ));
    if let Some(o) = s.outcome {
        out.push_str(&format!(" | outcome {o:?}"));
    }
    if let Some(b) = s.boss {
        let off = if s.boss_disabled { " (disabled)" } else { "" };
        out.push_str(&format!(" | boss {b:?}{off}"));
    }
    out.push_str(&format!(
        "\nscore {} / target {} | money ${} | hands {} discards {}\n",
        s.round_score, s.target, s.money, s.hands_left, s.discards_left
    ));
    out.push_str("jokers:");
    if s.jokers.is_empty() {
        out.push_str(" (none)");
    }
    for (i, j) in s.jokers.iter().enumerate() {
        out.push_str(&format!(" [{i}] {:?} (${})", j.id, j.sell_value));
    }
    out.push('\n');
    if !s.consumables.is_empty() {
        out.push_str("consumables:");
        for (i, c) in s.consumables.iter().enumerate() {
            let neg = if c.negative { " Negative" } else { "" };
            out.push_str(&format!(" [{i}] {:?}{neg} (${})", c.id, c.sell_value));
        }
        out.push('\n');
    }
    if s.phase == Phase::Shop {
        out.push_str(&shop_text(s));
    }
    if let Some(p) = &s.pack {
        out.push_str(&format!(
            "pack {:?} {:?}, {} pick(s) left:",
            p.spec.kind, p.spec.size, p.picks_left
        ));
        for (i, it) in p.items.iter().enumerate() {
            match it {
                Some(balatro_core::packs::PackItem::Consumable(id)) => {
                    out.push_str(&format!(" [{i}] {id:?}"))
                }
                Some(balatro_core::packs::PackItem::Joker(j)) => {
                    out.push_str(&format!(" [{i}] {:?} {:?}", j.id, j.edition))
                }
                Some(balatro_core::packs::PackItem::Card(c)) => {
                    out.push_str(&format!(" [{i}] {}", card_str(c)))
                }
                None => {}
            }
        }
        out.push('\n');
    }
    if !s.tags.is_empty() {
        out.push_str(&format!("tags: {:?}\n", s.tags));
    }
    if !s.vouchers.is_empty() {
        out.push_str(&format!("vouchers: {:?}\n", s.vouchers));
    }
    if matches!(s.phase, Phase::Play | Phase::PackOpen) && !s.deck.hand.is_empty() {
        out.push_str("hand:");
        for (i, c) in s.deck.hand.iter().enumerate() {
            out.push_str(&format!(" {i}:{}", card_str(c)));
        }
        out.push_str(&format!(" | draw pile {}\n", s.deck.draw_pile.len()));
    }
    out
}

fn report(info: &StepInfo) -> String {
    let mut out = String::new();
    if let Some(h) = &info.hand {
        out.push_str(&format!(
            "played {:?} (level {}): {} chips x {} mult = {}{}\n",
            h.hand_type,
            h.level,
            h.chips,
            h.mult,
            h.score,
            if h.invalid { " (rejected by boss)" } else { "" }
        ));
    }
    if let Some(t) = info.tag {
        out.push_str(&format!("took skip tag {t:?}\n"));
    }
    if let Some(p) = info.sold_for {
        out.push_str(&format!("sold for ${p}\n"));
    }
    if let Some(r) = &info.round {
        out.push_str(&format!(
            "round over: {} ({} / {})\n",
            if r.won { "blind beaten" } else { "LOST" },
            r.round_score,
            r.target
        ));
        if let Some(c) = &r.cash_out {
            out.push_str(&format!("cash out: {c:?}\n"));
        }
    }
    if info.money_before != info.money_after {
        out.push_str(&format!(
            "money ${} -> ${}\n",
            info.money_before, info.money_after
        ));
    }
    if info.phase_before != info.phase_after {
        out.push_str(&format!(
            "phase {:?} -> {:?}\n",
            info.phase_before, info.phase_after
        ));
    }
    out
}

/// Parse card positions into a bitmask over the current hand.
fn parse_mask(args: &[&str], hand_len: usize) -> Result<u16, String> {
    let mut mask: u16 = 0;
    for a in args {
        let i: usize = a.parse().map_err(|_| format!("not a position: {a:?}"))?;
        if i >= hand_len || i >= 16 {
            return Err(format!("position {i} out of range (hand has {hand_len})"));
        }
        mask |= 1 << i;
    }
    Ok(mask)
}

fn describe_error(e: &ActionError) -> String {
    format!("rejected: {e}")
}

fn do_action(sess: &mut Session, action: Action) -> String {
    match sess.game.step(action) {
        Ok(info) => {
            sess.replay.push(action);
            let mut out = report(&info);
            out.push_str(&summary(sess.game.state()));
            out
        }
        Err(e) => describe_error(&e),
    }
}

const HELP: &str = "commands: select | skip | play <pos..> | discard <pos..> | sell <i> | \
use <slot> [pos..] | sellc <slot> | buy <i> | buyuse <i> | pack <i> | pick <i> [pos..] | \
skippack | voucher <i> | reroll | rerollboss | move <from> <to> | end | legal | state | \
save <file> | load <file> | replay-save <file> | help | quit";

/// Handle one line. Returns `None` to quit, else the text to print.
fn handle(sess: &mut Session, line: &str) -> Option<String> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    let Some((&cmd, args)) = parts.split_first() else {
        return Some(String::new());
    };
    let hand_len = sess.game.state().deck.hand.len();
    let out = match cmd {
        "quit" | "exit" | "q" => return None,
        "help" | "?" => HELP.to_string(),
        "state" => summary(sess.game.state()),
        "select" => do_action(sess, Action::SelectBlind),
        "skip" => do_action(sess, Action::SkipBlind),
        "end" => do_action(sess, Action::EndShop),
        "play" | "discard" => match parse_mask(args, hand_len) {
            Ok(m) if cmd == "play" => do_action(sess, Action::Play(m)),
            Ok(m) => do_action(sess, Action::Discard(m)),
            Err(e) => format!("error: {e}"),
        },
        "sell" => match args {
            [a] => match a.parse::<u8>() {
                Ok(i) => do_action(sess, Action::SellJoker(i)),
                Err(_) => format!("error: not a joker slot: {a:?}"),
            },
            _ => "error: usage: sell <joker slot>".to_string(),
        },
        "use" => match args.split_first() {
            Some((a, rest)) => match a.parse::<u8>() {
                Ok(slot) => {
                    let targets = if rest.is_empty() {
                        Ok(0)
                    } else {
                        parse_mask(rest, hand_len)
                    };
                    match targets {
                        Ok(t) => do_action(sess, Action::UseConsumable { slot, targets: t }),
                        Err(e) => format!("error: {e}"),
                    }
                }
                Err(_) => format!("error: not a consumable slot: {a:?}"),
            },
            None => "error: usage: use <consumable slot> [hand positions..]".to_string(),
        },
        "buy" | "buyuse" | "pack" | "voucher" | "move" | "pick" => {
            let nums: Result<Vec<u8>, _> = args
                .iter()
                .take(if cmd == "move" { 2 } else { 1 })
                .map(|a| a.parse::<u8>())
                .collect();
            match (cmd, nums.as_deref()) {
                ("buy", Ok([i])) => do_action(sess, Action::Buy(*i)),
                ("buyuse", Ok([i])) => do_action(sess, Action::BuyAndUse(*i)),
                ("pack", Ok([i])) => do_action(sess, Action::OpenPack(*i)),
                ("voucher", Ok([i])) => do_action(sess, Action::BuyVoucher(*i)),
                ("move", Ok([a, b])) => do_action(sess, Action::MoveJoker { from: *a, to: *b }),
                ("pick", Ok([i])) => match parse_mask(&args[1..], hand_len) {
                    Ok(t) => do_action(
                        sess,
                        Action::PickPackItem {
                            index: *i,
                            targets: t,
                        },
                    ),
                    Err(e) => format!("error: {e}"),
                },
                _ => format!("error: bad arguments for {cmd}"),
            }
        }
        "skippack" => do_action(sess, Action::SkipPack),
        "reroll" => do_action(sess, Action::Reroll),
        "rerollboss" => do_action(sess, Action::RerollBoss),
        "sellc" => match args {
            [a] => match a.parse::<u8>() {
                Ok(i) => do_action(sess, Action::SellConsumable(i)),
                Err(_) => format!("error: not a consumable slot: {a:?}"),
            },
            _ => "error: usage: sellc <consumable slot>".to_string(),
        },
        "legal" => {
            let v = sess.game.legal_actions();
            let mut s = format!("{} legal actions", v.len());
            for a in v.iter().take(40) {
                s.push_str(&format!("\n  {a:?}"));
            }
            if v.len() > 40 {
                s.push_str(&format!("\n  ... and {} more", v.len() - 40));
            }
            s
        }
        "save" => match args {
            [f] => match std::fs::write(f, state_to_json(sess.game.state(), true)) {
                Ok(()) => format!("state saved to {f}"),
                Err(e) => format!("error: cannot write {f}: {e}"),
            },
            _ => "error: usage: save <file>".to_string(),
        },
        "load" => match args {
            [f] => match std::fs::read_to_string(f) {
                Ok(text) => match state_from_json(&text) {
                    Ok(state) => {
                        sess.game = Game::from_state(state);
                        sess.replay_valid = false;
                        format!("state loaded from {f}\n{}", summary(sess.game.state()))
                    }
                    Err(e) => format!("error: {e}"),
                },
                Err(e) => format!("error: cannot read {f}: {e}"),
            },
            _ => "error: usage: load <file>".to_string(),
        },
        "replay-save" => match args {
            [f] if !sess.replay_valid => {
                format!("error: cannot save {f}: a state was loaded, history is incomplete")
            }
            [f] => match std::fs::write(f, sess.replay.to_json()) {
                Ok(()) => format!(
                    "replay ({} actions) saved to {f}",
                    sess.replay.actions.len()
                ),
                Err(e) => format!("error: cannot write {f}: {e}"),
            },
            _ => "error: usage: replay-save <file>".to_string(),
        },
        other => format!("unknown command {other:?}; try `help`"),
    };
    Some(out)
}

fn parse_args() -> Result<u64, String> {
    let mut seed = 1u64;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--seed" => {
                let v = it.next().ok_or("--seed needs a value")?;
                seed = v.parse().map_err(|_| format!("bad seed {v:?}"))?;
            }
            "-h" | "--help" => {
                println!("usage: play_cli [--seed N]\n{HELP}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(seed)
}

fn main() {
    let seed = match parse_args() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };
    let mut sess = Session::new(seed);
    println!(
        "balatro-core {} play_cli, seed {seed}",
        balatro_core::version()
    );
    println!("{HELP}");
    print!("{}", summary(sess.game.state()));
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        println!("> {line}");
        match handle(&mut sess, &line) {
            Some(out) if out.is_empty() => {}
            Some(out) => println!("{}", out.trim_end()),
            None => break,
        }
        let _ = stdout.flush();
    }
}
