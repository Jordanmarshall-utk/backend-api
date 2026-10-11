from app.tools.hypothesis_scientific_guard import (
    clean_model_markup,
    guard_hypothesis_analysis,
    guard_hypothesis_synthesis,
)


def test_synthesis_normalizes_cha_formula_and_removes_arbitrary_target():
    text = (
        "A CHA+(PbI4)2/MAPbI3 interface will produce a 50% reduction in ion migration "
        "and at least 10% greater mobility."
    )
    guarded = guard_hypothesis_synthesis(text)
    assert "(CHA)\u2082PbI\u2084" in guarded
    assert "stoichiometry unresolved" in guarded
    assert "Here CHA denotes" not in guarded
    assert "CHA+(PbI4)2 notation is chemically ambiguous" in guarded
    assert "50%" not in guarded
    assert "10%" not in guarded
    assert "Chemical composition gate" in guarded
    assert "Quantitative-target gate" in guarded


def test_analysis_guard_adds_claim_appropriate_measurement_boundaries():
    report = (
        "The tests are scientifically sound and feasible. ToF-SIMS measures the ion migration "
        "rate, Hall effect measures mobility, and UV-Vis establishes interface band alignment. "
        "The proposed tests are well-aligned."
    )
    guarded = guard_hypothesis_analysis(
        report,
        "A stable CHA+(PbI4)2/MAPbI3 interface reduces ion motion by 50%.",
    )
    assert "Deterministic scientific audit" in guarded
    assert "not charge-balanced" in guarded
    assert "ToF-SIMS maps ion distributions" in guarded
    assert "mixed ionic-electronic conduction" in guarded
    assert "cannot establish buried-interface band alignment" in guarded
    assert "Definition-required" in guarded
    assert "50%" not in guarded


def test_undefined_three_apb_dj_assignment_requires_charge_definition():
    guarded = guard_hypothesis_synthesis(
        "Use 3-APB+ as a DJ spacer because its structure improves coupling."
    )
    assert "Chemical identity boundary" in guarded
    assert "two protonatable sites" in guarded
    assert "net 2+ charge" in guarded


def test_known_monocation_cannot_be_synthesized_as_a_diammonium_dj_spacer():
    guarded = guard_hypothesis_synthesis(
        "Use N-methyl-2-aminoethanol (NMEA+) as a diammonium DJ spacer to improve stability."
    )

    assert "DJ candidate-class gate" in guarded
    assert "one protonatable amine" in guarded
    assert "does not satisfy the requested diammonium DJ class" in guarded


def test_standalone_hash_delimiters_are_removed_recursively():
    value = {"messages": ["First\n\n##\n\nSecond", "## Heading"]}
    cleaned = clean_model_markup(value)
    assert cleaned["messages"][0] == "First\n\nSecond"
    assert cleaned["messages"][1] == "## Heading"


def test_synthesis_corrects_transport_direction_and_measurement_overreach():
    text = """
    The spacer suppresses ion migration by reducing the activation energy for ion migration.
    Time-resolved photoluminescence decay experiments will quantify the ion migration rate,
    and the decay rate constant reflects the rate of ion migration. XRD and SEM will evaluate
    crystallinity and grain boundary density. The effect is expected to translate into improved
    long-term device performance with a projected increase in power conversion efficiency (PCE).
    """
    guarded = guard_hypothesis_synthesis(text)
    lowered = guarded.lower()

    assert "reducing the activation energy for ion migration" not in lowered
    assert "increasing the effective activation barrier for ion migration" in lowered
    assert "trpl" not in lowered or "carrier recombination" in lowered
    assert "directly quantify grain-boundary density" in lowered
    assert "secondary outcome" in lowered
    assert "Mechanism and measurement gate" in guarded


def test_live_transport_wording_is_rewritten_in_the_hypothesis_body():
    generated = """
    A fluorinated XPA+ spacer suppresses ion migration by reducing the activation energy
    required for ion migration. Time-resolved photoluminescence decay will directly quantify
    the ion migration rate. The resulting reduction in migration is expected to translate
    into improved PCE.
    """

    guarded = guard_hypothesis_synthesis(generated)
    lowered = guarded.lower()

    assert "reducing the activation energy" not in lowered
    assert "increasing the effective activation barrier" in lowered
    assert "directly quantify" not in lowered
    assert "carrier recombination" in lowered
    assert "expected to translate into improved pce" not in lowered
    assert "secondary device-level outcome" in lowered
