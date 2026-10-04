use super::*;
use crate::bluff::ledger::ScriptLists;

/// Authored Hidden-phase context; native differential input is kept separate.
pub(crate) fn context() -> RevealContext {
    let roles = [
        (DataRole::Minion, CallbackRole::Minion),
        (DataRole::Confessor, CallbackRole::Confessor),
        (DataRole::Lover, CallbackRole::Lover),
        (DataRole::Hunter, CallbackRole::Hunter),
        (DataRole::Enlightened, CallbackRole::Enlightened),
    ];
    RevealContext {
        rule_version: SETUP_REVEAL_CALLBACKS_NATIVE_V5.into(),
        board_size: 5,
        trailer_mode: false,
        pools: SelectorPools {
            unique: vec!["Gemcrafter".into(), "Alchemist".into()],
            duplicate: vec![
                "Confessor".into(),
                "Lover".into(),
                "Hunter".into(),
                "Enlightened".into(),
            ],
            must_include: vec![],
            script: ScriptLists {
                villagers: vec![
                    "Confessor".into(),
                    "Lover".into(),
                    "Hunter".into(),
                    "Enlightened".into(),
                ],
                outcasts: vec![],
                minions: vec!["Minion".into()],
                demons: vec![],
            },
        },
        actors: roles
            .into_iter()
            .enumerate()
            .map(|(i, (data_role, action_role))| RevealActor {
                position: i as u8 + 1,
                data_role,
                action_role,
                runtime_evil: i == 0,
                bluff: BluffReference::Null,
                bluff_role: None,
                register_as: Some("Scout".into()),
                statuses: StatusState {
                    values: if i == 1 { vec![25] } else { vec![] },
                    resistance: vec![],
                    target_position: None,
                },
                remaining_continuations: 1,
                on_trigger_subscribed: false,
                character_start_acted: Some(false),
            })
            .collect(),
        resumes: (0..5)
            .map(|i| ResumeEvent {
                position: i + 1,
                resume_ordinal: u16::from(i),
                acquisition_ordinal: Some(u16::from(i)),
            })
            .collect(),
        spy_caches: BTreeMap::new(),
    }
}

#[test]
fn all_six_selector_outcomes_survive_with_conditional_service_weights() {
    let input = context();
    let paths = replay_reveal_callbacks(&input).unwrap();
    assert_eq!(paths.len(), 6);
    let names: BTreeSet<_> = paths
        .iter()
        .map(|p| p.trace[0].acquisition.as_ref().unwrap().bluff_role.as_str())
        .collect();
    assert_eq!(
        names,
        BTreeSet::from([
            "Confessor",
            "Lover",
            "Hunter",
            "Enlightened",
            "Gemcrafter",
            "Alchemist"
        ])
    );
    for path in paths {
        let selected = path.trace[0].acquisition.as_ref().unwrap();
        let unique = matches!(selected.bluff_role.as_str(), "Gemcrafter" | "Alchemist");
        assert_eq!(
            path.probability,
            Probability {
                numerator: if unique { 3 } else { 1 },
                denominator: 10
            }
        );
        assert_eq!(selected.rng_draw_count, 2);
        assert_eq!(selected.script_added, unique);
        assert_eq!(path.pools.duplicate, input.pools.duplicate);
        assert_eq!(path.pools.must_include, input.pools.must_include);
        assert_eq!(path.pools.unique, input.pools.unique);
        assert!(path.actors.iter().all(|a| a.register_as.is_none()
            && a.remaining_continuations == 0
            && a.character_start_acted == Some(false)));
        assert!(path
            .trace
            .iter()
            .flat_map(|t| &t.callbacks)
            .all(|c| c.trigger != Trigger::Start));
    }
}

#[test]
fn confessor_appearance_does_not_change_actual_truth_or_repeat_insertion() {
    let mut input = context();
    input.actors[1].statuses.target_position = Some(3);
    let path = replay_reveal_callbacks(&input)
        .unwrap()
        .into_iter()
        .find(|p| p.trace[0].acquisition.as_ref().unwrap().bluff_role == "Confessor")
        .unwrap();
    let minion = &path.actors[0];
    assert!(minion.is_lying());
    assert!(!minion.appears_lying());
    assert_eq!(minion.statuses.values, [25]);
    assert_eq!(path.trace[0].callbacks[0].dispatch, Dispatch::Act);
    let copied = &path.trace[0].callbacks[1];
    assert_eq!(
        (copied.role, copied.dispatch),
        (CallbackRole::Confessor, Dispatch::BluffAct)
    );
    assert!(copied.status_application.as_ref().unwrap().inserted);
    let repeated = path.trace[1].callbacks[0]
        .status_application
        .as_ref()
        .unwrap();
    assert!(repeated.accepted);
    assert!(!repeated.inserted);
    assert_eq!(repeated.target_after, None);
    assert_eq!(path.actors[1].statuses.values, [25]);
    assert_eq!(path.actors[1].statuses.target_position, None);
}

#[test]
fn copied_alchemist_non_day_dispatch_preserves_resistance_and_runtime_scope() {
    let mut input = context();
    input.actors[0].statuses.resistance = vec![25, 26];
    let path = replay_reveal_callbacks(&input)
        .unwrap()
        .into_iter()
        .find(|p| p.trace[0].acquisition.as_ref().unwrap().bluff_role == "Alchemist")
        .unwrap();
    assert!(path.actors[0].statuses.values.is_empty());
    assert_eq!(path.actors[0].statuses.resistance, [25, 26]);
    let copied: Vec<_> = path.trace[0]
        .callbacks
        .iter()
        .filter(|c| c.slot == RoleSlot::Bluff)
        .collect();
    assert_eq!(copied.len(), 2);
    assert!(copied.iter().all(|c| c.role == CallbackRole::Alchemist
        && c.dispatch == Dispatch::BluffAct
        && c.status_application.is_none()));
}

#[test]
fn folded_base_null_acquisitions_keep_provenance_without_rng_or_copied_slot() {
    let mut input = context();
    input.resumes.remove(0);
    let paths = replay_reveal_callbacks(&input).unwrap();
    assert_eq!(paths.len(), 1);
    let path = &paths[0];
    assert_eq!(
        path.probability,
        Probability {
            numerator: 1,
            denominator: 1
        }
    );
    assert_eq!(path.pools, input.pools);
    for trace in &path.trace {
        assert!(trace.event.acquisition_ordinal.is_some());
        assert!(trace.acquisition.is_none());
        assert_eq!(trace.previous_register_as.as_deref(), Some("Scout"));
        assert_eq!(trace.callbacks.len(), 2);
        assert!(trace.callbacks.iter().all(|c| c.slot == RoleSlot::Real));
    }
}

#[test]
fn new_domain_rejects_start_identity_mismatch_and_unmodeled_positive_support() {
    for mutation in 0..7 {
        let mut input = context();
        match mutation {
            0 => input.actors[0].statuses.values.push(30),
            1 => input.actors[0].runtime_evil = false,
            2 => input.actors[1].runtime_evil = true,
            3 => input.actors[1].action_role = CallbackRole::Alchemist,
            4 => input.pools.unique.push("Scout".into()),
            5 => input.actors[1].bluff_role = Some(CallbackRole::Confessor),
            6 => {
                input.actors[0].bluff = BluffReference::Live {
                    role: BluffRole::Alchemist,
                };
                input.actors[0].bluff_role = Some(CallbackRole::Confessor);
            }
            _ => unreachable!(),
        }
        let before = input.clone();
        assert_eq!(
            replay_reveal_callbacks(&input),
            Err(LedgerError::InvalidContext)
        );
        assert_eq!(input, before);
    }
}

#[test]
fn legacy_versions_do_not_gain_new_bluff_or_callback_support() {
    let mut input = context();
    input.rule_version = SETUP_CALLBACKS_NATIVE_V4.into();
    assert_eq!(
        replay_reveal_callbacks(&input),
        Err(LedgerError::InvalidContext)
    );
    input.resumes.clear();
    assert!(replay_reveal_callbacks(&input).is_ok());
    input.actors[0].bluff = BluffReference::Live {
        role: BluffRole::Alchemist,
    };
    input.actors[0].bluff_role = Some(CallbackRole::Alchemist);
    assert_eq!(
        replay_reveal_callbacks(&input),
        Err(LedgerError::InvalidContext)
    );
}

#[test]
fn old_data_roles_cannot_admit_new_bluffs_in_v1_through_v3() {
    for version in [
        REVEAL_CALLBACKS_NATIVE_V1,
        REVEAL_CALLBACKS_SPY_NATIVE_V2,
        REVEAL_CALLBACKS_START_NATIVE_V3,
    ] {
        let mut input = context();
        input.rule_version = version.into();
        input.resumes.clear();
        input.actors.truncate(1);
        input.actors[0].data_role = DataRole::TwinMinion;
        input.actors[0].action_role = CallbackRole::TwinMinion;
        input.actors[0].character_start_acted =
            (version == REVEAL_CALLBACKS_START_NATIVE_V3).then_some(false);
        input.pools.unique = vec!["Confessor".into()];
        input.pools.duplicate = vec!["Confessor".into()];
        assert!(replay_reveal_callbacks(&input).is_ok());
        input.actors[0].bluff = BluffReference::Live {
            role: BluffRole::Alchemist,
        };
        input.actors[0].bluff_role = Some(CallbackRole::Alchemist);
        assert_eq!(
            replay_reveal_callbacks(&input),
            Err(LedgerError::InvalidContext)
        );
    }
}
