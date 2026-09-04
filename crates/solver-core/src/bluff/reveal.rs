//! Bounded offline gameplay projection of delayed Character.Reveal callbacks.
//!
//! Starts at a caller-proven continuation resume and ends after AfterRoundStart.
//! Supports Lilis/Twin/Drunk data and Scout/Witness/Confessor bluff assets only.
//! It does not reconstruct coroutine order, native object graphs, view updates,
//! or subscribers. The caller must exclude intervening mutations, including
//! omitted resumes and view epilogues, from the modeled state. No live GameState
//! bridge consumes this module.

use super::ledger::{
    replay_selectors, LedgerError, Probability, Selector, SelectorEvent, SelectorLedger,
    SelectorPools, SelectorTrace, SELECTOR_LEDGER_NATIVE_V1,
};
use serde::{Deserialize, Serialize};

pub const REVEAL_CALLBACKS_NATIVE_V1: &str = "bounded_reveal_callbacks_native_v1";
const MAX_RESUMES: usize = 16;
const MAX_PATHS: usize = 65_536;
const MAX_ENTRIES: usize = 1_048_576;
const CORRUPTED: i32 = 10;
const APPEAR_TRUTHFUL: i32 = 25;
const APPEAR_LYING: i32 = 26;
const HEALTHY_BLUFF: i32 = 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataRole {
    Lilis,
    TwinMinion,
    Drunk,
}

/// These fieldless role classes have no Init/AfterRoundStart gameplay effect
/// except Confessor.OnInit. Keep the cloned action role separate from dataRef.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CallbackRole {
    Lilis,
    TwinMinion,
    Drunk,
    Scout,
    Witness,
    Confessor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BluffRole {
    Scout,
    Witness,
    Confessor,
}

impl BluffRole {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "Scout" => Some(Self::Scout),
            "Witness" => Some(Self::Witness),
            "Confessor" => Some(Self::Confessor),
            _ => None,
        }
    }
    fn callback(self) -> CallbackRole {
        match self {
            Self::Scout => CallbackRole::Scout,
            Self::Witness => CallbackRole::Witness,
            Self::Confessor => CallbackRole::Confessor,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum BluffReference {
    Null,
    Destroyed { role: BluffRole },
    Live { role: BluffRole },
}

impl BluffReference {
    fn is_live(self) -> bool {
        matches!(self, Self::Live { .. })
    }
}

/// Native ordered status/resistance lists and the single shared target pointer.
/// Other enum values are preserved, not interpreted as alternate lie flags.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StatusState {
    pub values: Vec<i32>,
    pub resistance: Vec<i32>,
    pub target_position: Option<u8>,
}

impl StatusState {
    fn apply(&mut self, status: i32, target: Option<u8>) -> StatusApplication {
        let accepted = !self.resistance.contains(&status);
        let inserted = accepted && !self.values.contains(&status);
        if inserted {
            self.values.push(status);
        }
        if accepted {
            self.target_position = target;
        }
        StatusApplication {
            status,
            accepted,
            inserted,
            target_after: self.target_position,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RevealActor {
    pub position: u8,
    pub data_role: DataRole,
    pub action_role: CallbackRole,
    pub runtime_evil: bool,
    pub bluff: BluffReference,
    pub bluff_role: Option<CallbackRole>,
    /// The supported data roles all overwrite this with native null.
    pub register_as: Option<String>,
    pub statuses: StatusState,
    pub remaining_continuations: u16,
    pub on_trigger_subscribed: bool,
}

impl RevealActor {
    pub fn is_lying(&self) -> bool {
        self.statuses.values.contains(&CORRUPTED)
            || (!self.statuses.values.contains(&HEALTHY_BLUFF)
                && (self.runtime_evil || self.bluff.is_live()))
    }

    pub fn appears_lying(&self) -> bool {
        self.statuses.values.contains(&APPEAR_LYING)
            || (!self.statuses.values.contains(&APPEAR_TRUTHFUL) && self.is_lying())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResumeEvent {
    pub position: u8,
    pub resume_ordinal: u16,
    /// Independent acquisition provenance. Must be absent when raw bluff is
    /// live; resume ordinal is never substituted for acquisition ordinal.
    pub acquisition_ordinal: Option<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RevealContext {
    pub rule_version: String,
    pub board_size: u8,
    pub trailer_mode: bool,
    pub pools: SelectorPools,
    pub actors: Vec<RevealActor>,
    pub resumes: Vec<ResumeEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StatusApplication {
    pub status: i32,
    pub accepted: bool,
    pub inserted: bool,
    pub target_after: Option<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Trigger {
    Init,
    AfterRoundStart,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Dispatch {
    Act,
    BluffAct,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RoleSlot {
    Real,
    Bluff,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CallbackTrace {
    pub trigger: Trigger,
    pub slot: RoleSlot,
    pub role: CallbackRole,
    pub dispatch: Dispatch,
    pub status_application: Option<StatusApplication>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResumeTrace {
    pub event: ResumeEvent,
    pub previous_register_as: Option<String>,
    pub acquisition: Option<SelectorTrace>,
    pub selector_status: Option<StatusApplication>,
    pub callbacks: Vec<CallbackTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RevealPath {
    pub probability: Probability,
    pub pools: SelectorPools,
    pub actors: Vec<RevealActor>,
    pub trace: Vec<ResumeTrace>,
}

fn selector_ledger(pools: SelectorPools, events: Vec<SelectorEvent>) -> SelectorLedger {
    SelectorLedger {
        rule_version: SELECTOR_LEDGER_NATIVE_V1.into(),
        pools,
        events,
    }
}

fn validate(context: &RevealContext) -> Result<(), LedgerError> {
    if context.rule_version != REVEAL_CALLBACKS_NATIVE_V1
        || context.board_size == 0
        || context.trailer_mode
        || context.resumes.len() > MAX_RESUMES
        || context.actors.len() > usize::from(context.board_size)
        || [
            &context.pools.unique,
            &context.pools.duplicate,
            &context.pools.must_include,
        ]
        .iter()
        .any(|pool| pool.iter().any(|role| BluffRole::parse(role).is_none()))
    {
        return Err(LedgerError::InvalidContext);
    }
    // Reuse the selector ledger's canonical asset, faction and occurrence caps.
    replay_selectors(&selector_ledger(context.pools.clone(), Vec::new()))?;
    let mut seen = [false; 256];
    for actor in &context.actors {
        if actor.position == 0
            || actor.position > context.board_size
            || seen[usize::from(actor.position)]
            || actor.on_trigger_subscribed
            || actor.statuses.values.contains(&HEALTHY_BLUFF)
            || actor.statuses.values.len() > 256
            || actor.statuses.resistance.len() > 256
            || actor
                .statuses
                .target_position
                .is_some_and(|p| p == 0 || p > context.board_size)
            || actor.register_as.as_ref().is_some_and(|role| {
                crate::knowledge_base::get_card(role).is_none_or(|card| card.name != role)
            })
        {
            return Err(LedgerError::InvalidContext);
        }
        seen[usize::from(actor.position)] = true;
    }
    let mut last_resume = None;
    let mut last_acquisition = None;
    for event in &context.resumes {
        if !seen[usize::from(event.position)]
            || last_resume.is_some_and(|last| last >= event.resume_ordinal)
            || event
                .acquisition_ordinal
                .is_some_and(|a| last_acquisition.is_some_and(|last| last >= a))
        {
            return Err(LedgerError::InvalidContext);
        }
        last_resume = Some(event.resume_ordinal);
        if event.acquisition_ordinal.is_some() {
            last_acquisition = event.acquisition_ordinal;
        }
    }
    Ok(())
}

fn callbacks(actor: &mut RevealActor) -> Vec<CallbackTrace> {
    let mut result = Vec::new();
    for trigger in [Trigger::Init, Trigger::AfterRoundStart] {
        // Character.Act computes this once before real then copied dispatch.
        let lying = actor.is_lying();
        let real_dispatch = if !lying || (actor.runtime_evil && actor.bluff_role.is_some()) {
            Dispatch::Act
        } else {
            Dispatch::BluffAct
        };
        for (slot, role, dispatch) in [
            (RoleSlot::Real, Some(actor.action_role), real_dispatch),
            (
                RoleSlot::Bluff,
                actor.bluff_role,
                if lying {
                    Dispatch::BluffAct
                } else {
                    Dispatch::Act
                },
            ),
        ] {
            if let Some(role) = role {
                let status_application =
                    if trigger == Trigger::Init && role == CallbackRole::Confessor {
                        Some(actor.statuses.apply(APPEAR_TRUTHFUL, None))
                    } else {
                        None
                    };
                result.push(CallbackTrace {
                    trigger,
                    slot,
                    role,
                    dispatch,
                    status_application,
                });
            }
        }
    }
    result
}

fn push_bounded(
    paths: &mut Vec<RevealPath>,
    path: RevealPath,
    entries: &mut usize,
) -> Result<(), LedgerError> {
    let script = &path.pools.script;
    let count = path.pools.unique.len()
        + path.pools.duplicate.len()
        + path.pools.must_include.len()
        + script.villagers.len()
        + script.outcasts.len()
        + script.minions.len()
        + script.demons.len()
        + path
            .actors
            .iter()
            .map(|a| 1 + a.statuses.values.len() + a.statuses.resistance.len())
            .sum::<usize>()
        + path
            .trace
            .iter()
            .map(|t| 1 + t.callbacks.len())
            .sum::<usize>();
    *entries = entries.checked_add(count).ok_or(LedgerError::Capacity)?;
    if paths.len() >= MAX_PATHS || *entries > MAX_ENTRIES {
        return Err(LedgerError::Capacity);
    }
    paths.push(path);
    Ok(())
}

/// Compose proven resumes through the synchronous gameplay callbacks. Repeated
/// bodies consume another pending continuation but do not reacquire a live bluff.
/// Any unsupported branch rejects the whole invocation; no partial worlds leak.
pub fn replay_reveal_callbacks(context: &RevealContext) -> Result<Vec<RevealPath>, LedgerError> {
    validate(context)?;
    let mut paths = vec![RevealPath {
        probability: Probability {
            numerator: 1,
            denominator: 1,
        },
        pools: context.pools.clone(),
        actors: context.actors.clone(),
        trace: Vec::new(),
    }];
    for event in &context.resumes {
        let mut next = Vec::new();
        let mut entries = 0;
        for mut path in paths {
            let actor_index = path
                .actors
                .iter()
                .position(|a| a.position == event.position)
                .ok_or(LedgerError::InvalidContext)?;
            let actor = &mut path.actors[actor_index];
            if actor.remaining_continuations == 0
                || actor.bluff.is_live() == event.acquisition_ordinal.is_some()
            {
                return Err(LedgerError::InvalidContext);
            }
            actor.remaining_continuations -= 1;
            let previous_register_as = actor.register_as.take();
            if let Some(acquisition_ordinal) = event.acquisition_ordinal {
                let selector = match actor.data_role {
                    DataRole::Lilis => Selector::Demon,
                    DataRole::TwinMinion => Selector::Minion,
                    DataRole::Drunk => Selector::Drunk {
                        corruption_resistant: actor.statuses.resistance.contains(&CORRUPTED),
                    },
                };
                let selector_status = if actor.data_role == DataRole::Drunk {
                    Some(actor.statuses.apply(CORRUPTED, Some(actor.position)))
                } else {
                    None
                };
                let draws = replay_selectors(&selector_ledger(
                    path.pools.clone(),
                    vec![SelectorEvent {
                        position: event.position,
                        acquisition_ordinal,
                        selector,
                    }],
                ))?;
                for draw in draws {
                    let mut branch = path.clone();
                    branch.probability = branch
                        .probability
                        .multiply(draw.probability.numerator, draw.probability.denominator)?;
                    branch.pools = draw.pools;
                    let acquisition = draw
                        .trace
                        .into_iter()
                        .next()
                        .ok_or(LedgerError::InvalidContext)?;
                    let role = BluffRole::parse(&acquisition.bluff_role)
                        .ok_or(LedgerError::InvalidContext)?;
                    let actor = &mut branch.actors[actor_index];
                    actor.bluff = BluffReference::Live { role };
                    actor.bluff_role = Some(role.callback());
                    let callbacks = callbacks(actor);
                    branch.trace.push(ResumeTrace {
                        event: event.clone(),
                        previous_register_as: previous_register_as.clone(),
                        acquisition: Some(acquisition),
                        selector_status: selector_status.clone(),
                        callbacks,
                    });
                    push_bounded(&mut next, branch, &mut entries)?;
                }
            } else {
                let callbacks = callbacks(actor);
                path.trace.push(ResumeTrace {
                    event: event.clone(),
                    previous_register_as,
                    acquisition: None,
                    selector_status: None,
                    callbacks,
                });
                push_bounded(&mut next, path, &mut entries)?;
            }
        }
        paths = next;
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bluff::ledger::ScriptLists;

    fn names(roles: &[&str]) -> Vec<String> {
        roles.iter().map(|r| r.to_string()).collect()
    }

    fn actor(position: u8, data_role: DataRole) -> RevealActor {
        RevealActor {
            position,
            data_role,
            action_role: match data_role {
                DataRole::Lilis => CallbackRole::Lilis,
                DataRole::TwinMinion => CallbackRole::TwinMinion,
                DataRole::Drunk => CallbackRole::Drunk,
            },
            runtime_evil: data_role == DataRole::Lilis,
            bluff: BluffReference::Null,
            bluff_role: None,
            register_as: Some("Bard".into()),
            statuses: StatusState {
                values: vec![],
                resistance: vec![],
                target_position: None,
            },
            remaining_continuations: 2,
            on_trigger_subscribed: false,
        }
    }

    fn resume(position: u8, resume_ordinal: u16, acquisition_ordinal: Option<u16>) -> ResumeEvent {
        ResumeEvent {
            position,
            resume_ordinal,
            acquisition_ordinal,
        }
    }

    fn context(actors: Vec<RevealActor>, resumes: Vec<ResumeEvent>) -> RevealContext {
        RevealContext {
            rule_version: REVEAL_CALLBACKS_NATIVE_V1.into(),
            board_size: 8,
            trailer_mode: false,
            pools: SelectorPools {
                unique: names(&["Witness"]),
                duplicate: names(&["Scout"]),
                must_include: vec![],
                script: ScriptLists {
                    villagers: names(&["Bard"]),
                    outcasts: names(&["Drunk"]),
                    minions: names(&["Twin Minion"]),
                    demons: names(&["Lilis"]),
                },
            },
            actors,
            resumes,
        }
    }

    #[test]
    fn repeated_twin_resume_acquires_once_but_repeats_both_callback_passes() {
        let input = context(
            vec![actor(4, DataRole::TwinMinion)],
            vec![resume(4, 2, Some(0)), resume(4, 8, None)],
        );
        let paths = replay_reveal_callbacks(&input).unwrap();
        assert_eq!(paths.len(), 2);
        assert_eq!(
            paths.iter().map(|p| p.probability).collect::<Vec<_>>(),
            vec![
                Probability {
                    numerator: 2,
                    denominator: 5
                },
                Probability {
                    numerator: 3,
                    denominator: 5
                }
            ]
        );
        for path in &paths {
            assert_eq!(path.actors[0].remaining_continuations, 0);
            assert_eq!(path.actors[0].register_as, None);
            assert_eq!(path.trace[0].previous_register_as.as_deref(), Some("Bard"));
            assert_eq!(path.trace[1].previous_register_as, None);
            assert!(path.trace[1].acquisition.is_none());
            assert_eq!(path.trace[0].callbacks, path.trace[1].callbacks);
            assert_eq!(
                path.trace[0]
                    .callbacks
                    .iter()
                    .map(|c| (c.trigger, c.slot))
                    .collect::<Vec<_>>(),
                vec![
                    (Trigger::Init, RoleSlot::Real),
                    (Trigger::Init, RoleSlot::Bluff),
                    (Trigger::AfterRoundStart, RoleSlot::Real),
                    (Trigger::AfterRoundStart, RoleSlot::Bluff)
                ]
            );
            assert!(path.trace[0]
                .callbacks
                .iter()
                .all(|c| c.dispatch == Dispatch::BluffAct));
        }
        assert_eq!(input.actors[0].remaining_continuations, 2);
        assert_eq!(input.actors[0].bluff, BluffReference::Null);
    }

    #[test]
    fn mixed_acquisitions_compose_pool_mass_and_ordinal_provenance() {
        let mut input = context(
            vec![actor(1, DataRole::TwinMinion), actor(2, DataRole::Lilis)],
            vec![
                resume(1, 0, Some(0)),
                resume(2, 7, Some(3)),
                resume(1, 10, None),
            ],
        );
        input.pools.must_include = names(&["Scout", "Confessor"]);
        let paths = replay_reveal_callbacks(&input).unwrap();
        assert_eq!(paths.len(), 4);
        assert_eq!(
            paths.iter().map(|p| p.probability).collect::<Vec<_>>(),
            vec![
                Probability {
                    numerator: 1,
                    denominator: 5
                },
                Probability {
                    numerator: 1,
                    denominator: 5
                },
                Probability {
                    numerator: 3,
                    denominator: 10
                },
                Probability {
                    numerator: 3,
                    denominator: 10
                }
            ]
        );
        for path in paths {
            assert_eq!(
                path.trace[1]
                    .acquisition
                    .as_ref()
                    .unwrap()
                    .event
                    .acquisition_ordinal,
                3
            );
            assert_eq!(path.trace[1].event.resume_ordinal, 7);
            assert_eq!(path.trace[2].acquisition, None);
        }
    }

    #[test]
    fn drunk_then_confessor_clears_shared_target_without_changing_actual_truth() {
        let mut input = context(
            vec![actor(1, DataRole::Drunk)],
            vec![resume(1, 0, Some(0)), resume(1, 1, None)],
        );
        input.pools.unique = names(&["Confessor"]);
        let path = replay_reveal_callbacks(&input).unwrap().remove(0);
        assert_eq!(
            path.actors[0].statuses.values,
            vec![CORRUPTED, APPEAR_TRUTHFUL]
        );
        assert_eq!(
            path.trace[0].selector_status,
            Some(StatusApplication {
                status: CORRUPTED,
                accepted: true,
                inserted: true,
                target_after: Some(1),
            })
        );
        assert_eq!(
            path.trace[0].callbacks[1].status_application,
            Some(StatusApplication {
                status: APPEAR_TRUTHFUL,
                accepted: true,
                inserted: true,
                target_after: None,
            })
        );
        assert_eq!(path.trace[1].selector_status, None);
        assert_eq!(
            path.trace[1].callbacks[1]
                .status_application
                .as_ref()
                .unwrap()
                .inserted,
            false
        );
        assert_eq!(path.actors[0].statuses.target_position, None);
        assert!(path.actors[0].is_lying());
        assert!(!path.actors[0].appears_lying());
        assert!(path
            .trace
            .iter()
            .flat_map(|t| &t.callbacks)
            .all(|c| c.dispatch == Dispatch::BluffAct));
    }

    #[test]
    fn resistance_preserves_existing_status_membership_and_target() {
        let mut body = actor(1, DataRole::Drunk);
        body.statuses = StatusState {
            values: vec![CORRUPTED, APPEAR_TRUTHFUL, APPEAR_LYING],
            resistance: vec![CORRUPTED, APPEAR_TRUTHFUL],
            target_position: Some(7),
        };
        let mut input = context(vec![body.clone()], vec![resume(1, 1, Some(4))]);
        input.pools.unique = names(&["Confessor"]);
        let path = replay_reveal_callbacks(&input).unwrap().remove(0);
        assert_eq!(path.actors[0].statuses, body.statuses);
        assert!(!path.trace[0].selector_status.as_ref().unwrap().accepted);
        assert!(
            !path.trace[0].callbacks[1]
                .status_application
                .as_ref()
                .unwrap()
                .accepted
        );
        assert!(path.actors[0].appears_lying()); // AppearLying beats AppearTruthful
    }

    #[test]
    fn already_corrupted_drunk_retargets_self_before_resisted_confessor() {
        let mut body = actor(2, DataRole::Drunk);
        body.statuses = StatusState {
            values: vec![CORRUPTED],
            resistance: vec![APPEAR_TRUTHFUL],
            target_position: Some(7),
        };
        let mut input = context(vec![body], vec![resume(2, 0, Some(0))]);
        input.pools.unique = names(&["Confessor"]);
        let path = replay_reveal_callbacks(&input).unwrap().remove(0);
        assert_eq!(path.actors[0].statuses.values, vec![CORRUPTED]);
        assert_eq!(path.actors[0].statuses.target_position, Some(2));
        assert!(!path.trace[0].selector_status.as_ref().unwrap().inserted);
        assert!(path.trace[0].selector_status.as_ref().unwrap().accepted);
    }

    #[test]
    fn resisted_clean_drunk_still_lies_from_live_bluff() {
        let mut body = actor(1, DataRole::Drunk);
        body.statuses.resistance = vec![CORRUPTED];
        let input = context(vec![body], vec![resume(1, 0, Some(0))]);
        let path = replay_reveal_callbacks(&input).unwrap().remove(0);
        assert!(!path.actors[0].statuses.values.contains(&CORRUPTED));
        assert!(path.actors[0].is_lying());
        assert!(path.trace[0]
            .callbacks
            .iter()
            .all(|c| c.dispatch == Dispatch::BluffAct));
    }

    #[test]
    fn live_bluff_skips_selection_and_preserves_independent_copied_role() {
        let mut body = actor(1, DataRole::Drunk);
        body.bluff = BluffReference::Live {
            role: BluffRole::Scout,
        };
        body.bluff_role = Some(CallbackRole::Confessor);
        let mut input = context(vec![body], vec![resume(1, 0, None)]);
        input.pools.unique.clear(); // would fail if Drunk incorrectly acquired again
        let path = replay_reveal_callbacks(&input).unwrap().remove(0);
        assert_eq!(path.pools, input.pools);
        assert_eq!(path.actors[0].bluff, input.actors[0].bluff);
        assert_eq!(path.actors[0].bluff_role, Some(CallbackRole::Confessor));
        assert_eq!(path.actors[0].statuses.values, vec![APPEAR_TRUTHFUL]);
        assert_eq!(path.trace[0].acquisition, None);
        assert_eq!(path.trace[0].selector_status, None);
    }

    #[test]
    fn destroyed_bluff_reacquires_and_overwrites_stale_copied_role() {
        let mut body = actor(1, DataRole::Lilis);
        body.bluff = BluffReference::Destroyed {
            role: BluffRole::Scout,
        };
        body.bluff_role = Some(CallbackRole::Confessor);
        let input = context(vec![body], vec![resume(1, 0, Some(0))]);
        let path = replay_reveal_callbacks(&input).unwrap().remove(0);
        assert_eq!(
            path.actors[0].bluff,
            BluffReference::Live {
                role: BluffRole::Witness
            }
        );
        assert_eq!(path.actors[0].bluff_role, Some(CallbackRole::Witness));
        assert!(path.actors[0].statuses.values.is_empty());
        assert_eq!(path.trace[0].callbacks[0].dispatch, Dispatch::Act);
        assert_eq!(path.trace[0].callbacks[1].dispatch, Dispatch::BluffAct);
    }

    #[test]
    fn evil_real_dispatch_depends_on_copied_pointer_not_appearance() {
        let mut body = actor(1, DataRole::Lilis);
        body.bluff = BluffReference::Live {
            role: BluffRole::Scout,
        };
        body.bluff_role = None;
        body.statuses.values = vec![CORRUPTED, APPEAR_TRUTHFUL];
        let mut input = context(vec![body], vec![resume(1, 0, None)]);
        let without = replay_reveal_callbacks(&input).unwrap().remove(0);
        assert_eq!(without.trace[0].callbacks.len(), 2);
        assert!(without.trace[0]
            .callbacks
            .iter()
            .all(|c| c.dispatch == Dispatch::BluffAct));
        input.actors[0].bluff_role = Some(CallbackRole::Scout);
        let with = replay_reveal_callbacks(&input).unwrap().remove(0);
        assert_eq!(with.trace[0].callbacks.len(), 4);
        assert_eq!(with.trace[0].callbacks[0].dispatch, Dispatch::Act);
        assert_eq!(with.trace[0].callbacks[1].dispatch, Dispatch::BluffAct);
        assert!(!with.actors[0].appears_lying());
    }

    #[test]
    fn data_selector_and_shared_action_role_are_not_conflated() {
        let mut body = actor(1, DataRole::Lilis);
        body.action_role = CallbackRole::Confessor;
        let input = context(vec![body], vec![resume(1, 0, Some(0))]);
        let path = replay_reveal_callbacks(&input).unwrap().remove(0);
        assert_eq!(
            path.trace[0].acquisition.as_ref().unwrap().event.selector,
            Selector::Demon
        );
        assert_eq!(path.trace[0].callbacks[0].role, CallbackRole::Confessor);
        assert_eq!(path.actors[0].statuses.values, vec![APPEAR_TRUTHFUL]);
    }

    #[test]
    fn order_and_continuation_provenance_are_required() {
        let base = context(vec![actor(1, DataRole::Lilis)], vec![resume(1, 0, Some(0))]);
        for resumes in [
            vec![resume(1, 0, None)],
            vec![resume(2, 0, Some(0))],
            vec![resume(1, 0, Some(0)), resume(1, 0, None)],
            vec![resume(1, 0, Some(0)), resume(1, 1, Some(1))],
            vec![
                resume(1, 0, Some(0)),
                resume(1, 1, None),
                resume(1, 2, None),
            ],
        ] {
            assert_eq!(
                replay_reveal_callbacks(&RevealContext {
                    resumes,
                    ..base.clone()
                }),
                Err(LedgerError::InvalidContext)
            );
        }
        let mut input = base.clone();
        input.actors[0].remaining_continuations = 0;
        assert_eq!(
            replay_reveal_callbacks(&input),
            Err(LedgerError::InvalidContext)
        );
        input = base;
        input.actors.push(actor(2, DataRole::Drunk));
        input.resumes.push(resume(2, 1, Some(0)));
        assert_eq!(
            replay_reveal_callbacks(&input),
            Err(LedgerError::InvalidContext)
        );
    }

    #[test]
    fn unsupported_hooks_and_assets_reject_wholesale() {
        let base = context(vec![actor(1, DataRole::Lilis)], vec![resume(1, 0, Some(0))]);
        let mut input = base.clone();
        input.actors[0].on_trigger_subscribed = true;
        assert_eq!(
            replay_reveal_callbacks(&input),
            Err(LedgerError::InvalidContext)
        );
        input = base.clone();
        input.actors[0].statuses.values.push(HEALTHY_BLUFF);
        assert_eq!(
            replay_reveal_callbacks(&input),
            Err(LedgerError::InvalidContext)
        );
        input = base.clone();
        input.pools.unique.push("Rambler".into());
        assert_eq!(
            replay_reveal_callbacks(&input),
            Err(LedgerError::InvalidContext)
        );
        input = base;
        input.trailer_mode = true;
        assert_eq!(
            replay_reveal_callbacks(&input),
            Err(LedgerError::InvalidContext)
        );
    }

    #[test]
    fn failing_branch_never_returns_successful_siblings() {
        let mut input = context(
            vec![actor(1, DataRole::TwinMinion), actor(2, DataRole::Lilis)],
            vec![resume(1, 0, Some(0)), resume(2, 1, Some(1))],
        );
        input.pools.unique.clear();
        input.pools.must_include = names(&["Scout"]);
        // Twin duplicate preserves Scout for Lilis, but unique consumes it.
        assert_eq!(
            replay_reveal_callbacks(&input),
            Err(LedgerError::EmptySupport)
        );
        assert_eq!(input.actors[0].bluff, BluffReference::Null);
    }

    #[test]
    fn bounded_paths_and_schema_fail_closed() {
        let base = context(vec![actor(1, DataRole::Lilis)], vec![resume(1, 0, Some(0))]);
        let serialized = serde_json::to_string(&base).unwrap();
        assert_eq!(
            serde_json::from_str::<RevealContext>(&serialized).unwrap(),
            base
        );
        let mut value = serde_json::to_value(&base).unwrap();
        value["actors"][0]["data_role"] = "spy".into();
        assert!(serde_json::from_value::<RevealContext>(value).is_err());
        let mut value = serde_json::to_value(&base).unwrap();
        value["live_solver_input"] = true.into();
        assert!(serde_json::from_value::<RevealContext>(value).is_err());
        let mut input = base.clone();
        input.resumes = vec![resume(1, 0, None); 17];
        assert_eq!(
            replay_reveal_callbacks(&input),
            Err(LedgerError::InvalidContext)
        );
        input = base;
        input.pools.unique = vec!["Scout".into(); 2000];
        assert_eq!(replay_reveal_callbacks(&input), Err(LedgerError::Capacity));
    }
}
