//! Phase 2C: scripted `play_cli` sessions through stdin.

use balatro_core::serialize::{Replay, state_from_json};
use std::io::Write;
use std::process::{Command, Stdio};

fn run(args: &[&str], script: &str) -> (String, bool) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_play_cli"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn play_cli");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(script.as_bytes())
        .expect("write script");
    let out = child.wait_with_output().expect("wait");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        out.status.success(),
    )
}

#[test]
fn scripted_session_plays_a_hand() {
    let (out, ok) = run(
        &["--seed", "1"],
        "select\nplay 0 1 2 3 4\nlegal\ndiscard 0\nquit\n",
    );
    assert!(ok);
    assert!(out.contains("phase BlindSelect -> Play"), "{out}");
    assert!(out.contains("hand:"), "{out}");
    assert!(out.contains("chips x"), "{out}");
    assert!(out.contains("legal actions"), "{out}");
}

#[test]
fn bad_commands_do_not_crash() {
    let script = "bogus\nplay\nplay 0\nsell 0\nend\nselect\nplay 99\nplay x\nplay 0 1 2 3 4 5\n\
                  discard\nsell\nsave\nload /nonexistent/file\nskip\nquit\n";
    let (out, ok) = run(&["--seed", "3"], script);
    assert!(ok, "{out}");
    assert!(out.contains("unknown command"), "{out}");
    assert!(out.contains("rejected:"), "{out}");
    assert!(out.contains("out of range"), "{out}");
    assert!(out.contains("not a position"), "{out}");
    assert!(out.contains("cannot read"), "{out}");
}

#[test]
fn bad_arguments_exit_with_error() {
    let (_, ok) = run(&["--seed", "abc"], "");
    assert!(!ok);
    let (_, ok) = run(&["--nope"], "");
    assert!(!ok);
}

#[test]
fn save_load_and_replay_save() {
    let dir = std::env::temp_dir().join(format!("balatro_cli_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmp dir");
    let state = dir.join("state.json");
    let replay = dir.join("replay.json");
    let script = format!(
        "select\nplay 0 1\nsave {s}\nreplay-save {r}\ndiscard 0\nload {s}\nreplay-save {r}.2\nstate\nquit\n",
        s = state.display(),
        r = replay.display()
    );
    let (out, ok) = run(&["--seed", "7"], &script);
    assert!(ok, "{out}");
    assert!(out.contains("state loaded"), "{out}");
    assert!(out.contains("history is incomplete"), "{out}");

    let s = state_from_json(&std::fs::read_to_string(&state).unwrap()).unwrap();
    let r = Replay::from_json(&std::fs::read_to_string(&replay).unwrap()).unwrap();
    assert_eq!(r.seed, 7);
    assert_eq!(r.actions.len(), 2);
    assert_eq!(r.play().unwrap().state(), &s);
    std::fs::remove_dir_all(&dir).ok();
}
