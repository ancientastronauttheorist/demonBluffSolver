//! Bounded native Reveal presentation tail, after gameplay callbacks.
//! Assumes valid required UI objects/assets and inert Unity lifecycle callbacks.
//! Projects identity sources and UI writes, not pixels or asset getter internals.
use super::ledger::LedgerError;
use super::reveal::BluffReference;
use super::twin_writer::BodyState;
use serde::{Deserialize, Serialize};

pub const REVEAL_VIEW_NATIVE_V1: &str = "reveal_view_native_v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RevealViewContext {
    pub rule_version: String,
    pub raw_bluff: BluffReference,
    pub body: BodyState,
    pub pickable_active: bool,
    pub rip_active: bool,
    /// None means Unity-null (absent or destroyed) optional disguise icon.
    pub disguise_icon_active: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VisualSource {
    CurrentData,
    RawBluff,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ViewWrite {
    Colors {
        source: VisualSource,
    },
    Refresh {
        created_dead: bool,
        pickable_write: Option<bool>,
        rip_write: Option<bool>,
        disguise_write: Option<bool>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RevealViewResult {
    pub context: RevealViewContext,
    /// Reveal chooses name/art from raw bluff alone, before UpdateView chooses
    /// background/border identity using state and the separate revealed flag.
    pub name_art_source: VisualSource,
    pub final_color_source: VisualSource,
    pub writes: Vec<ViewWrite>,
}

fn refresh(context: &mut RevealViewContext, live_bluff: bool, writes: &mut Vec<ViewWrite>) {
    let pickable_write = (context.body.pickable_uses < 1).then_some(false);
    if let Some(active) = pickable_write {
        context.pickable_active = active;
    }
    let created_dead = context.body.state == 20 && !context.body.created_dead_presentation;
    let rip_write = created_dead.then_some(true);
    if created_dead {
        context.body.created_dead_presentation = true;
        context.rip_active = true;
    }
    // killedByDemon preserves the icon's old active state; it does not hide it.
    let disguise_write = if context.disguise_icon_active.is_some() && !context.body.killed_by_demon
    {
        Some(matches!(context.body.state, 20 | 30) && live_bluff)
    } else {
        None
    };
    if let Some(active) = disguise_write {
        context.disguise_icon_active = Some(active);
    }
    writes.push(ViewWrite::Refresh {
        created_dead,
        pickable_write,
        rip_write,
        disguise_write,
    });
}

pub fn replay_reveal_view(input: &RevealViewContext) -> Result<RevealViewResult, LedgerError> {
    if input.rule_version != REVEAL_VIEW_NATIVE_V1 {
        return Err(LedgerError::InvalidContext);
    }
    let mut context = input.clone();
    let live = matches!(context.raw_bluff, BluffReference::Live { .. });
    let name_art_source = if live {
        VisualSource::RawBluff
    } else {
        VisualSource::CurrentData
    };
    let final_color_source =
        if !live || matches!(context.body.state, 20 | 30) || context.body.revealed {
            VisualSource::CurrentData
        } else {
            VisualSource::RawBluff
        };
    let mut writes = Vec::new();
    // No-bluff: RevealReal -> UpdateViewReal -> RefreshView.
    // Live-bluff: UpdateView -> RefreshView, then explicit RefreshView.
    writes.push(ViewWrite::Colors {
        source: if live {
            final_color_source
        } else {
            VisualSource::CurrentData
        },
    });
    refresh(&mut context, live, &mut writes);
    if live {
        refresh(&mut context, live, &mut writes);
    }
    // Both paths end with UpdateView -> RefreshView.
    writes.push(ViewWrite::Colors {
        source: final_color_source,
    });
    refresh(&mut context, live, &mut writes);
    Ok(RevealViewResult {
        context,
        name_art_source,
        final_color_source,
        writes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bluff::reveal::BluffRole;
    fn input() -> RevealViewContext {
        RevealViewContext {
            rule_version: REVEAL_VIEW_NATIVE_V1.into(),
            raw_bluff: BluffReference::Live {
                role: BluffRole::Scout,
            },
            body: BodyState {
                state: 5,
                previous_state: 10,
                revealed: false,
                killed_by_demon: false,
                pickable_uses: 1,
                acted_info_count: 3,
                created_dead_presentation: false,
                on_state_change_subscribed: false,
            },
            pickable_active: false,
            rip_active: false,
            disguise_icon_active: Some(true),
        }
    }
    #[test]
    fn live_bluff_runs_three_refreshes_without_revealing_hidden_body() {
        let original = input();
        let result = replay_reveal_view(&original).unwrap();
        assert_eq!(result.writes.len(), 5);
        assert_eq!(result.name_art_source, VisualSource::RawBluff);
        assert_eq!(result.final_color_source, VisualSource::RawBluff);
        assert_eq!(result.context.body, original.body);
        assert_eq!(result.context.disguise_icon_active, Some(false));
        assert!(!result.context.pickable_active);
    }
    #[test]
    fn null_and_destroyed_bluffs_take_two_refresh_real_path() {
        for raw in [
            BluffReference::Null,
            BluffReference::Destroyed {
                role: BluffRole::Scout,
            },
        ] {
            let mut original = input();
            original.raw_bluff = raw;
            let result = replay_reveal_view(&original).unwrap();
            assert_eq!(result.writes.len(), 4);
            assert_eq!(result.name_art_source, VisualSource::CurrentData);
            assert_eq!(result.final_color_source, VisualSource::CurrentData);
        }
    }
    #[test]
    fn death_presentation_is_created_once_despite_repeated_refreshes() {
        let mut original = input();
        original.body.state = 20;
        let result = replay_reveal_view(&original).unwrap();
        assert!(result.context.body.created_dead_presentation);
        assert!(result.context.rip_active);
        assert_eq!(
            result
                .writes
                .iter()
                .filter(|w| matches!(
                    w,
                    ViewWrite::Refresh {
                        created_dead: true,
                        ..
                    }
                ))
                .count(),
            1
        );
        let second = replay_reveal_view(&result.context).unwrap();
        assert!(second.writes.iter().all(|w| !matches!(
            w,
            ViewWrite::Refresh {
                created_dead: true,
                ..
            }
        )));
        let mut expected = original.body;
        expected.created_dead_presentation = true;
        assert_eq!(result.context.body, expected);
    }
    #[test]
    fn existing_dead_presentation_does_not_force_rip_active() {
        let mut original = input();
        original.body.state = 20;
        original.body.created_dead_presentation = true;
        let result = replay_reveal_view(&original).unwrap();
        assert!(!result.context.rip_active);
        assert!(result.writes.iter().all(|w| !matches!(
            w,
            ViewWrite::Refresh {
                rip_write: Some(_),
                ..
            }
        )));
    }
    #[test]
    fn color_identity_and_disguise_use_different_revealed_predicates() {
        for (state, revealed, disguise) in [(20, false, true), (30, false, true), (5, true, false)]
        {
            let mut original = input();
            original.body.state = state;
            original.body.revealed = revealed;
            let result = replay_reveal_view(&original).unwrap();
            assert_eq!(result.name_art_source, VisualSource::RawBluff);
            assert_eq!(result.final_color_source, VisualSource::CurrentData);
            assert_eq!(result.context.disguise_icon_active, Some(disguise));
        }
    }
    #[test]
    fn demon_kill_preserves_optional_icon_but_does_not_skip_death_creation() {
        for icon in [None, Some(false), Some(true)] {
            let mut original = input();
            original.body.state = 20;
            original.body.killed_by_demon = true;
            original.disguise_icon_active = icon;
            let result = replay_reveal_view(&original).unwrap();
            assert_eq!(result.context.disguise_icon_active, icon);
            assert!(result.context.body.created_dead_presentation);
        }
    }
    #[test]
    fn exhausted_pickable_is_hidden_but_positive_uses_preserve_active_state() {
        for uses in [-1, 0, 1, 4] {
            for active in [false, true] {
                let mut original = input();
                original.body.pickable_uses = uses;
                original.pickable_active = active;
                let result = replay_reveal_view(&original).unwrap();
                assert_eq!(result.context.pickable_active, uses > 0 && active);
                assert_eq!(result.context.body.pickable_uses, uses);
            }
        }
    }
    #[test]
    fn projection_is_versioned_and_rejects_unknown_fields() {
        let mut original = input();
        original.rule_version = "guessed".into();
        assert_eq!(
            replay_reveal_view(&original),
            Err(LedgerError::InvalidContext)
        );
        let mut json = serde_json::to_value(input()).unwrap();
        json["missing_objects_ok"] = true.into();
        assert!(serde_json::from_value::<RevealViewContext>(json).is_err());
    }
}
