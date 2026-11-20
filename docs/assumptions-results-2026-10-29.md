# Assumptions generation pilot — 2026-10-29

This is a first implementation trial for Rita's proposed Assumptions task, not
a frozen team benchmark. All samples were generated and judged with
`gpt-6-sol`; the same-model judge is not independent validation. The six
batch samples used three interdisciplinary topics, twice each, with two
speakers, one target assumption, `max_refilms=1`, and
`OPENAI_MIN_OUTPUT_TOKENS=2048`. Raw JSON and run logs are under
`data/outputs/`.

| Scenario | Batch run | Transcript ID | Final QA | Status | Main failure |
|---|---:|---|---|---|---|
| Green space / wellbeing | 1 | `506cd302` | failed | missing | Broader goal and dependent decision unsupported |
| Green space / wellbeing | 2 | `aa043eb2` | failed | missing | Decision dependence unsupported |
| Urban heat / health | 1 | `cb7c01a8` | failed | resolved | Decision changed to cool roofs; premise challenged |
| Urban heat / health | 2 | `1f1422db` | failed | missing | Required decision and evidence chain missing |
| Outdoor learning | 1 | `c98a1bf3` | failed | exposed | Participant raised comparability of contribution counts |
| Outdoor learning | 2 | `64adb818` | failed | exposed | Participant raised the target measurement gap |

**Automatic QA pass rate: 0/6.** Final status counts: missing 3,
exposed 2, resolved 1. All six scored 2/5 on naturalness, below the
unchanged 4/5 threshold. These are generator outcomes, not manually
adjudicated validity labels. Hadley's group should inspect the saved
transcripts, quoted evidence, inference, and every QA attempt before using
any record as benchmark data.

Before the batch, one complete green-space rehearsal (`19332fa3`) failed as
exposed, with a missing joint site-selection decision. An earlier rehearsal
(`f548b6f5`) stopped at turn 4 when a short leak-check request exhausted
`max_output_tokens`; its log is saved as `interrupted_f548b6f5.log`.
The first intended batch green-space run (`93f19d6b`) failed during Agent 1
because the model returned a blank required ground-truth field. Its run log
is retained as `batch_green_1.log`; it produced no transcript. The curator
now retries malformed assumptions answers, and `aa043eb2` is the replacement
second complete green-space sample.

The transcript-only identification pilot is conditional on at least one
candidate passing QA. No failed sample will be represented as eligible.

## Supplemental candidate attempt

A separate three-speaker rain-garden run (`809c192a`) produced an implicit
target with recoverability 4/5 and implicitness 5/5, but failed naturalness
at 2/5. QA cited repeated inlet/overflow caveats and polished, complete
arguments. It is not part of the six-sample pass rate and is not eligible
for the identification pilot. The next supplemental run tests shorter,
less repetitive turns without changing the rubric.

The second three-speaker run (`630feb01`) also scored 4/5 recoverability and
5/5 implicitness after re-filming, but 2/5 naturalness, so it remains a failure.
A third three-speaker run (`f1bdf5d9`), after shortening assumptions turns,
scored 3/5 recoverability, 4/5 implicitness, and 2/5 naturalness. It also
remains a failure. Across the six planned samples and three supplemental
samples, no transcript passed the unchanged full QA gate. Consequently the
required identification pilot on a **qualified** candidate could not be run.
A transcript-only **diagnostic** on `809c192a` is saved as
`data/outputs/diagnostic_pilot_809c192a.json`. The identification model
received only the numbered dialogue and a generic task. It identified a
different plausible premise about map completeness rather than the seeded
infiltration-to-flooding premise. This illustrates possible answer ambiguity;
it is not a qualifying pilot or a model-ability finding. The same model was
used to generate, judge, and identify.
