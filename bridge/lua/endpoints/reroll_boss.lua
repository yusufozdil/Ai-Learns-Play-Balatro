---RerollBoss endpoint (Director's Cut / Retcon), part of the ai-plays-balatro bridge patch.
return {
  name = "reroll_boss",
  description = "Reroll the boss blind of the current ante (Director's Cut once per ante, Retcon unlimited; costs $10)",
  schema = {},
  requires_state = { G.STATES.BLIND_SELECT },
  execute = function(_, send_response)
    local used = G.GAME.used_vouchers or {}
    local allowed = used["v_retcon"] or (used["v_directors_cut"] and not G.GAME.round_resets.boss_rerolled)
    if not allowed then
      send_response({
        message = "Boss reroll not available (needs Director's Cut unused this ante, or Retcon)",
        name = BB_ERROR_NAMES.NOT_ALLOWED,
      })
      return
    end
    if (G.GAME.dollars - G.GAME.bankrupt_at) - 10 < 0 then
      send_response({
        message = "Not enough dollars to reroll the boss ($10)",
        name = BB_ERROR_NAMES.NOT_ALLOWED,
      })
      return
    end
    if not (G.blind_select_opts and G.blind_select_opts.boss) then
      send_response({ message = "No boss blind pane", name = BB_ERROR_NAMES.INVALID_STATE })
      return
    end
    local before = G.GAME.round_resets.blind_choices.Boss
    G.FUNCS.reroll_boss({})
    G.E_MANAGER:add_event(Event({
      trigger = "condition",
      blocking = false,
      func = function()
        local done = G.STATE == G.STATES.BLIND_SELECT
          and not G.CONTROLLER.locks.boss_reroll
          and G.blind_select_opts
          and G.blind_select_opts.boss
          and G.GAME.round_resets.blind_choices.Boss ~= nil
          and (G.GAME.round_resets.blind_choices.Boss ~= before or G.GAME.round_resets.boss_rerolled)
        if done then
          send_response(BB_GAMESTATE.get_gamestate())
        end
        return done
      end,
    }))
  end,
}
