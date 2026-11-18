---ContinueEndless endpoint, part of the council-of-jokers bridge patch.
---After the ante-8 boss is beaten the game shows the win screen; its "Endless" button is
---`G.FUNCS.exit_overlay_menu`. This endpoint waits for the win screen, presses that button and
---returns the gamestate (still ROUND_EVAL; `cash_out` follows, but only once
---`bridge.cash_out_ready` is true: the end-of-round rows are built after the overlay closes,
---a `cash_out` before that crashes the game, see docs/design/bridge.md "Endless mode").
return {
  name = "continue_endless",
  description = "Dismiss the win screen and keep playing in endless mode",
  schema = {},
  requires_state = { G.STATES.ROUND_EVAL },
  execute = function(_, send_response)
    if not G.GAME.won then
      send_response({ message = "The run has not been won", name = BB_ERROR_NAMES.NOT_ALLOWED })
      return
    end
    local seen_overlay = false
    local pressed = false
    local waited = 0
    G.E_MANAGER:add_event(Event({
      trigger = "condition",
      blocking = false,
      blockable = false,
      created_on_pause = true,
      func = function()
        waited = waited + 1
        if G.OVERLAY_MENU and not pressed then
          seen_overlay = true
          pressed = true
          G.FUNCS.exit_overlay_menu()
          return false
        end
        local finished = pressed and not G.OVERLAY_MENU
        -- The win screen is created by the game's end-of-round events, which can take many
        -- seconds after BalatroBot's `play` response (long scoring animations). Wait up to ~90 s
        -- (5400 frames). The round-eval rows are only built after the overlay closes (the game
        -- is paused while it is open); the caller polls `bridge.cash_out_ready` before `cash_out`.
        if not seen_overlay and waited > 5400 then
          send_response({ message = "win screen never appeared", name = BB_ERROR_NAMES.NOT_ALLOWED })
          return true
        end
        if finished then
          send_response(BB_GAMESTATE.get_gamestate())
        end
        return finished
      end,
    }))
  end,
}
