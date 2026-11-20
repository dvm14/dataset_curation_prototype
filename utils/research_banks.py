"""
research_banks.py
Bank of research backgrounds for the meeting generator.

Each entry is a realistic interdisciplinary research problem tied to the
disciplines that would work on it, plus a short note on what each field
cares about and the words it uses. Before this bank, every task ran on one
hard-coded problem (urban green space and mental health), so every meeting
sounded like the same project.

Usage (main.py does this when CONFIG["research_problem"] is None):
    entry = pick_research(disciplines=None, n=3, distance="distant")
    text  = research_text(entry)       # goes into Agent 1 as research_problem
    entry["disciplines"]               # the disciplines to cast
"""

import random

# distance: "close" = neighbouring fields, "distant" = far apart,
#           "mixed" = three or more fields with both near and far pairs.
RESEARCH_BANK = [
    # ── Two disciplines, close ────────────────────────────
    {
        "id": "sleep_adolescent_emotion",
        "distance": "close",
        "disciplines": ["neuroscience", "psychology"],
        "problem": (
            "Understanding how chronic sleep loss changes emotion regulation "
            "in teenagers, and whether later school start times help"
        ),
        "lenses": {
            "neuroscience": "amygdala-prefrontal connectivity, fMRI and actigraphy, "
                            "sample size and scan-to-scan variability",
            "psychology": "emotion-regulation questionnaires, diary studies, "
                          "validity of self-report, developmental stages",
        },
    },
    {
        "id": "tidal_wetland_restoration",
        "distance": "close",
        "disciplines": ["ecology", "environmental engineering"],
        "problem": (
            "Deciding how to restore a degraded tidal marsh so it both recovers "
            "native vegetation and reduces storm-surge damage to nearby homes"
        ),
        "lenses": {
            "ecology": "species composition, sediment and salinity gradients, "
                       "recovery trajectories, reference sites",
            "environmental engineering": "wave attenuation, design storms, "
                                         "hydrodynamic models, permitting and cost",
        },
    },
    {
        "id": "pollinator_strips",
        "distance": "close",
        "disciplines": ["ecology", "biology"],
        "problem": (
            "Working out why wild bee numbers on farms did not recover after "
            "wildflower strips were planted along field edges"
        ),
        "lenses": {
            "ecology": "landscape context, habitat connectivity, nesting "
                       "resources, multi-year survey design",
            "biology": "bee physiology and foraging range, pesticide "
                       "sublethal effects, species identification",
        },
    },
    {
        "id": "wearable_arrhythmia",
        "distance": "close",
        "disciplines": ["computer science", "biomedical engineering"],
        "problem": (
            "Building a wrist-worn sensor and model that flags irregular "
            "heart rhythm early enough for a clinic to follow up"
        ),
        "lenses": {
            "computer science": "training data, false-positive rate, model "
                                "drift, on-device compute and battery",
            "biomedical engineering": "signal quality and motion artifact, "
                                      "sensor placement, regulatory pathway",
        },
    },
    {
        "id": "school_vaccination_outreach",
        "distance": "close",
        "disciplines": ["epidemiology", "public health"],
        "problem": (
            "Planning how to raise school-entry vaccination rates in a county "
            "that just had a measles cluster"
        ),
        "lenses": {
            "epidemiology": "susceptible pockets, case definitions, "
                            "surveillance lag, vaccine coverage thresholds",
            "public health": "clinic hours, trusted messengers, school nurse "
                             "workload, outreach budget",
        },
    },
    # ── Two disciplines, distant ──────────────────────────
    {
        "id": "consciousness_end_of_life",
        "distance": "distant",
        "disciplines": ["neuroscience", "philosophy"],
        "problem": (
            "Deciding whether brain-imaging evidence of awareness in "
            "unresponsive patients should count in end-of-life decisions"
        ),
        "lenses": {
            "neuroscience": "command-following paradigms, EEG and fMRI "
                            "signatures, false negatives, patient variability",
            "philosophy": "what counts as awareness, moral status, "
                          "inference from evidence to a normative claim",
        },
    },
    {
        "id": "mangrove_payments",
        "distance": "distant",
        "disciplines": ["ecology", "economics"],
        "problem": (
            "Designing payments to coastal landowners to keep mangroves "
            "standing instead of converting them to shrimp ponds"
        ),
        "lenses": {
            "ecology": "carbon storage and fish nursery value, "
                       "regeneration time, tipping points",
            "economics": "opportunity cost, additionality, enforcement, "
                         "willingness to accept",
        },
    },
    {
        "id": "heart_failure_readmission",
        "distance": "distant",
        "disciplines": ["clinical medicine", "sociology"],
        "problem": (
            "Reducing 30-day hospital readmissions for older adults with "
            "heart failure after they go home"
        ),
        "lenses": {
            "clinical medicine": "randomized trials, medication adherence, "
                                 "readmission as an outcome, guideline-directed therapy",
            "sociology": "household and caregiving arrangements, trust in "
                         "the health system, interviews and observation",
        },
    },
    {
        "id": "chw_blood_pressure",
        "distance": "distant",
        "disciplines": ["clinical medicine", "sociology"],
        "problem": (
            "Evaluating a program where community health workers visit "
            "patients with uncontrolled high blood pressure at home"
        ),
        "lenses": {
            "clinical medicine": "blood pressure control as the outcome, "
                                 "randomized or stepped-wedge designs, "
                                 "medication titration, follow-up visits",
            "sociology": "trust between workers and families, who counts as "
                         "a patient, home visits as observed practice, "
                         "interviews with workers and patients",
        },
    },
    {
        "id": "walk_in_diabetes_clinic",
        "distance": "distant",
        "disciplines": ["clinical medicine", "sociology"],
        "problem": (
            "Deciding whether a nurse-led walk-in diabetes clinic at a "
            "safety-net hospital is working and for whom"
        ),
        "lenses": {
            "clinical medicine": "HbA1c change, comparison groups, "
                                 "no-show and follow-up rates, hospital "
                                 "quality metrics",
            "sociology": "why patients come at odd hours, clinic culture "
                         "and staff roles, stigma, qualitative accounts "
                         "of waiting and being seen",
        },
    },
    {
        "id": "ai_math_tutor",
        "distance": "distant",
        "disciplines": ["computer science", "education"],
        "problem": (
            "Evaluating an AI math tutor used in middle-school classrooms "
            "and deciding what it should and should not do for students"
        ),
        "lenses": {
            "computer science": "model accuracy, hint generation, logging "
                                "and privacy, latency in school networks",
            "education": "formative assessment, productive struggle, "
                         "teacher workload, classroom norms",
        },
    },
    {
        "id": "rural_point_of_care",
        "distance": "distant",
        "disciplines": ["biomedical engineering", "anthropology"],
        "problem": (
            "Understanding why a low-cost point-of-care blood test is "
            "adopted in some rural clinics and left in a drawer in others"
        ),
        "lenses": {
            "biomedical engineering": "assay sensitivity, power and cold-chain "
                                      "limits, calibration and maintenance",
            "anthropology": "ethnographic fieldwork, local ideas of illness, "
                            "who gets to decide, informal referral networks",
        },
    },
    {
        "id": "wastewater_surveillance",
        "distance": "distant",
        "disciplines": ["epidemiology", "science & technology studies"],
        "problem": (
            "Setting up wastewater monitoring for infectious disease in a "
            "mid-sized city and deciding who sees the data"
        ),
        "lenses": {
            "epidemiology": "lead time over clinical cases, normalization, "
                            "catchment definition, detection limits",
            "science & technology studies": "who counts as an expert, how "
                                            "data becomes a public fact, "
                                            "infrastructure and trust",
        },
    },
    {
        "id": "scarcity_decisions",
        "distance": "distant",
        "disciplines": ["neuroscience", "economics"],
        "problem": (
            "Testing whether financial stress narrows how people weigh "
            "present against future payoffs, and what that means for "
            "program design"
        ),
        "lenses": {
            "neuroscience": "stress hormones, reward and valuation circuits, "
                            "lab tasks versus real life",
            "economics": "discounting, incentive-compatible tasks, "
                         "field experiments, welfare measures",
        },
    },
    # ── Three disciplines ─────────────────────────────────
    {
        "id": "green_space_mental_health",
        "distance": "distant",
        "disciplines": ["environmental science", "public health", "sociology"],
        "problem": (
            "Understanding how urban green space affects mental health "
            "outcomes in low-income communities"
        ),
        "lenses": {
            "environmental science": "tree canopy and land-cover mapping, "
                                     "buffer distances, seasonal change",
            "public health": "exposure measurement, comparison "
                             "neighborhoods, survey and stress-biomarker design",
            "sociology": "neighborhood trust, displacement, who uses a park, "
                         "community partnerships",
        },
    },
    {
        "id": "wildfire_smoke_workers",
        "distance": "mixed",
        "disciplines": ["environmental science", "epidemiology", "sociology"],
        "problem": (
            "Protecting outdoor workers from wildfire smoke when "
            "air-quality alerts arrive after the shift has already started"
        ),
        "lenses": {
            "environmental science": "PM2.5 sensors, smoke plume forecasts, "
                                     "indoor-outdoor infiltration",
            "epidemiology": "short-term exposure and respiratory visits, "
                            "confounding by heat, case definitions",
            "sociology": "informal and day labor, employer incentives, "
                         "language access, fear of losing shifts",
        },
    },
    {
        "id": "community_fisheries",
        "distance": "mixed",
        "disciplines": ["ecology", "economics", "anthropology"],
        "problem": (
            "Redesigning catch limits for a community-managed reef fishery "
            "so stocks recover without pushing families out of fishing"
        ),
        "lenses": {
            "ecology": "stock assessment, spawning aggregations, bycatch, "
                       "reef condition surveys",
            "economics": "quota design, income diversification, "
                         "compliance incentives, price variation",
            "anthropology": "customary tenure, kinship and sharing norms, "
                            "who is counted as a fisher, trust in managers",
        },
    },
    {
        "id": "clinic_triage_model",
        "distance": "mixed",
        "disciplines": ["computer science", "epidemiology", "sociology"],
        "problem": (
            "Introducing a risk-scoring model to help community clinics "
            "decide which patients to call first"
        ),
        "lenses": {
            "computer science": "calibration, subgroup performance, "
                                "feature leakage, explaining scores",
            "epidemiology": "outcome definition, selection into care, "
                            "validation in a new population",
            "sociology": "how staff actually use scores, accountability, "
                         "patient perception of being ranked",
        },
    },
    {
        "id": "classroom_neurofeedback",
        "distance": "mixed",
        "disciplines": ["neuroscience", "education", "biomedical engineering"],
        "problem": (
            "Judging whether a headband neurofeedback program improves "
            "attention in elementary classrooms"
        ),
        "lenses": {
            "neuroscience": "EEG attention markers, placebo and expectation "
                            "effects, durability of change",
            "education": "learning outcomes, teacher buy-in, lesson "
                         "time, equity of access",
            "biomedical engineering": "dry-electrode signal quality, "
                                      "child fit, device safety, battery",
        },
    },
    {
        "id": "heat_resilient_housing",
        "distance": "mixed",
        "disciplines": ["environmental engineering", "public health", "economics"],
        "problem": (
            "Choosing which retrofits to fund so older apartment buildings "
            "stay safe during multi-day heat waves"
        ),
        "lenses": {
            "environmental engineering": "indoor temperature modeling, "
                                         "insulation and cooling options, "
                                         "grid load",
            "public health": "heat illness risk by age, cooling-center use, "
                             "emergency visits",
            "economics": "cost per avoided illness, landlord incentives, "
                         "who pays the electric bill",
        },
    },
    {
        "id": "forest_carbon_credits",
        "distance": "mixed",
        "disciplines": ["ecology", "political science", "science & technology studies"],
        "problem": (
            "Assessing whether forest carbon credits actually protect "
            "forest and who gets to decide how that is measured"
        ),
        "lenses": {
            "ecology": "baselines, leakage, forest inventory and remote "
                       "sensing, permanence",
            "political science": "governance and enforcement, land "
                                 "tenure, state versus local authority",
            "science & technology studies": "how a number becomes a credit, "
                                            "auditors and certifiers, "
                                            "whose knowledge counts",
        },
    },
]


def pick_research(disciplines=None, n=None, distance=None):
    """
    Choose a research background.

    disciplines — if given, return an entry whose disciplines match them
                  (same set first, then any entry that contains them all).
                  Returns None when nothing fits; the caller keeps its own
                  problem text.
    n           — number of speakers wanted, when disciplines is None
    distance    — "close" / "distant" / "mixed", preferred but not forced
    """
    if disciplines:
        wanted = set(disciplines)
        exact = [e for e in RESEARCH_BANK if set(e["disciplines"]) == wanted]
        sub = [e for e in RESEARCH_BANK if wanted <= set(e["disciplines"])]
        pool = exact or sub
        return random.choice(pool) if pool else None

    pool = [e for e in RESEARCH_BANK if n is None or len(e["disciplines"]) == n]
    # Distance is a preference. It is dropped when it would leave too few
    # choices (e.g. 3 speakers with "distant" would otherwise always give
    # the same background).
    by_distance = [e for e in pool if distance is None or e["distance"] == distance]
    pool = by_distance if len(by_distance) >= 3 else (pool or RESEARCH_BANK)
    return random.choice(pool)


def research_text(entry, with_lenses=True, only=None):
    """
    The text handed to Agent 1 as the research problem.
    only — list of disciplines actually in the meeting; lenses for any
           other field in the entry are left out.
    """
    text = entry["problem"]
    if with_lenses:
        lenses = "; ".join(f"{d}: {note}" for d, note in entry["lenses"].items()
                           if only is None or d in only)
        text += f". What each field brings: {lenses}"
    return text
