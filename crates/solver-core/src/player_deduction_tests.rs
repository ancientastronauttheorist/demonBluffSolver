use super::*;

fn world(position: u8) -> Scenario {
    let mut scenario = Scenario::default();
    scenario.evil_positions.insert(position, "Baa".into());
    scenario
}

fn backend() -> (Vec<Scenario>, SolverResult) {
    let candidates: Vec<_> = (1..=4).map(world).collect();
    let result = SolverResult {
        definite_evil: vec![2],
        definite_good: vec![1, 3, 4],
        bombardier_positions: vec![],
        n_scenarios: 4,
        n_surviving: 1,
        surviving_scenarios: vec![world(2)],
        reasoning: vec![],
    };
    (candidates, result)
}

fn assert_invariant(candidates: &[Scenario], result: &SolverResult) {
    assert!(matches!(
        check_backend(4, candidates, result),
        Err(PlayerDeductionOutcome::Incomplete {
            kind: IncompleteKind::BackendInvariant,
            ..
        })
    ));
}

#[test]
fn missing_world_cannot_hide_behind_a_matching_candidate_count() {
    let (mut candidates, result) = backend();
    candidates[3] = world(1);
    assert_invariant(&candidates, &result);
}

#[test]
fn candidate_and_survivor_residual_state_fail_closed() {
    let (mut candidates, mut result) = backend();
    candidates[0].corrupted.insert(3);
    assert_invariant(&candidates, &result);
    candidates[0] = world(1);
    result.surviving_scenarios[0].puppet_position = Some(3);
    assert_invariant(&candidates, &result);
    result.surviving_scenarios[0] = world(2);
    result.surviving_scenarios[0]
        .pre_twin_current_roles
        .insert(1, "Hunter".into());
    assert_invariant(&candidates, &result);
}

#[test]
fn counts_duplicates_wrong_roles_and_bounds_fail_closed() {
    let (candidates, result) = backend();
    let mut changed = result.clone();
    changed.n_scenarios = 3;
    assert_invariant(&candidates, &changed);
    changed = result.clone();
    changed.n_surviving = 2;
    assert_invariant(&candidates, &changed);
    changed.surviving_scenarios.push(world(2));
    assert_invariant(&candidates, &changed);
    changed = result.clone();
    changed.surviving_scenarios[0] = world(5);
    assert_invariant(&candidates, &changed);
    changed.surviving_scenarios[0] = world(2);
    changed.surviving_scenarios[0]
        .evil_positions
        .insert(2, "Imp".into());
    assert_invariant(&candidates, &changed);
}

#[test]
fn backend_cannot_supply_unjustified_definites_or_role_actions() {
    let (candidates, result) = backend();
    let mut changed = result.clone();
    changed.definite_evil = vec![1];
    assert_invariant(&candidates, &changed);
    changed = result.clone();
    changed.definite_good = vec![1, 2, 3, 4];
    assert_invariant(&candidates, &changed);
    changed = result;
    changed.bombardier_positions = vec![1];
    assert_invariant(&candidates, &changed);
}

#[test]
fn contradiction_has_no_vacuous_definite_positions() {
    let (candidates, mut result) = backend();
    result.n_surviving = 0;
    result.surviving_scenarios.clear();
    result.definite_evil.clear();
    result.definite_good.clear();
    assert_eq!(check_backend(4, &candidates, &result), Ok(BTreeSet::new()));
    result.definite_good = vec![1, 2, 3, 4];
    assert_invariant(&candidates, &result);
}

#[test]
fn canonical_world_order_does_not_depend_on_backend_order() {
    let (mut candidates, mut result) = backend();
    candidates.reverse();
    result.surviving_scenarios = vec![world(3), world(1)];
    result.n_surviving = 2;
    result.definite_evil.clear();
    result.definite_good = vec![2, 4];
    assert_eq!(
        check_backend(4, &candidates, &result),
        Ok(BTreeSet::from([
            ConditionalWorld { baa_position: 1 },
            ConditionalWorld { baa_position: 3 },
        ]))
    );
}
