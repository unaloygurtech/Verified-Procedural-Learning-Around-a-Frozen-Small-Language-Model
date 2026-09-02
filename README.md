# Verified Procedural Learning Around a Frozen Small Language Model

A completed 20-experiment study of whether a frozen small language model can acquire reusable capability through external, verified procedural state.

## Publication status

**Status:** Complete and archived  
**Scope:** Research prototype and reproducible experiment record  
**Experiments:** 0001–0020  
**Central result:** Positive systems result; negative result for the original continual-learning hypothesis

### Final conclusion

> **Verified Procedural Learning Around a Frozen Small Language Model technically produced a working system, but it did not discover the targeted original continual-learning architecture.**

The original hypothesis was that a small frozen language model could become generally more capable through accumulated external experience. Experiments 0015–0017 showed that the model could not reliably convert novel documentation or verified examples into new semantic contracts, executable programs, or canonical intermediate representations.

Experiments 0018–0020 established a different, well-defined result: bounded program search, external verification, behavioral canonicalization, persistence, retrieval, composition, and reuse can form a reliable procedural-memory system without changing the base model.

This repository is preserved as a finished research result. It is not an active product roadmap.

## Verification status

- 20 frozen experiment reports are included, covering Experiments 0001–0020.
- 28 deterministic `unittest` modules are included, containing **144 test methods** in the current source tree and experiment coverage through Experiment 0020.
- CLI parser coverage includes every runnable experiment command from 003 through 020; the persistence suite includes dedicated Experiment 0020 tests.
- A historical frozen Docker run reports **138 tests passing**; this is kept separate from the current source-tree count.
- GitHub Actions runs the deterministic test suite on every push and pull request.
- Key numerical results are summarized in this README and preserved in the corresponding reports: 12/12 candidate coverage, 20/20 canonicalized acquisition, 32/32 persistent acquisition, 48/48 final transfer, 0 wrong activations, and 0/6 false canonical merges.

## Quick start

### Requirements

- Docker Desktop or Docker Engine with Compose v2.
- Internet access on the first run so Docker can pull the runtime image and the model.
- Enough free disk space for the container images, model cache, and generated runtime data.
- The default configuration runs the model on CPU. GPU use is optional and host/runtime dependent; no specific VRAM amount is required by the default setup.
- The documented 4096-token context is a reproducibility parameter, not a host-specific setting.

### First run

From a cloned checkout:

```bash
git clone https://github.com/unaloygurtech/Verified-Procedural-Learning-Around-a-Frozen-Small-Language-Model.git
cd Verified-Procedural-Learning-Around-a-Frozen-Small-Language-Model
cp .env.example .env
docker compose up --build -d
```

On Windows PowerShell, use `Copy-Item .env.example .env` instead of `cp .env.example .env`.

Then check the services:

```bash
docker compose ps
docker compose exec air-core python -m air_core.cli doctor
```

The first run may take longer than five minutes because the container image and model are downloaded. Subsequent starts are usually much faster when the image and model cache are already present. The repository does not promise a fixed installation time because network speed, disk speed, and Docker cache state vary.

## Research question

Can a fixed small language model gain reusable capability from experience while its base parameters remain frozen?

The study separated two claims that are often conflated:

1. **System-level external capability growth:** a surrounding system acquires and reuses verified procedures.
2. **Base-model cognitive capability growth:** the frozen language model itself becomes better at novel semantic tasks.

The experiments support the first claim only within a bounded task grammar. They do not support the second claim.

## Related work and experiment history

See [Related work](docs/RELATED_WORK.md) for the academic context and [Experiment timeline](docs/EXPERIMENT-TIMELINE.md) for the progression from 0001 through 0020.

## Implemented architecture

```text
verified examples
        ↓
bounded candidate generation and search
        ↓
static safety and public validation
        ↓
hidden and edge validation
        ↓
behavioral canonicalization
        ↓
persistent executable procedural state
        ↓
retrieval, composition, and model-free execution
```

The learned state is an external procedural library, not a model-weight update. Once an artifact is learned and verified, the hot path can retrieve and execute it without another model call.

The repository contains:

- typed executable artifacts;
- bounded program induction;
- objective validation gates;
- hidden and edge tests before activation;
- behavioral canonicalization;
- provenance and immutable artifact handling;
- persistent state with integrity checks;
- indexed and faceted retrieval;
- bounded composition and transfer;
- deterministic tests and containerized execution.

## Experiment summary

| Experiment | Research question | Main result |
|---|---|---|
| 0001 | Can a compact verified skill beat raw experience? | Manual verified skill: 7/8; 65.7% fewer prompt tokens than raw experience |
| 0002 | Can skills be consolidated automatically? | Bounded symbolic search: 8/8; unconstrained model strategies failed |
| 0003 | Can multiple skill families be discovered and routed? | Automated learned skills: 9/12 held-out |
| 0004 | Can same-schema procedures be distinguished? | Content discrimination worked; prose compression lost semantics |
| 0005 | Does a typed executable representation improve reliability? | Typed executable synthesis: 12/12 held-out |
| 0006 | Can verified skills be composed? | Learned composition: 9/9; impossible plans were safely rejected |
| 0007 | Can a capability gap trigger a missing skill? | Bounded missing-skill path: 9/9 after a supplied primitive |
| 0008 | Can a bounded Python/API capability be acquired? | Learned artifact: 12/12; model-only: 0/12 |
| 0009 | Does a frozen generic Python learner generalize? | Activation: 2/8; wrong activation: 0 |
| 0010 | Can an opaque unseen API be learned from documentation? | 3/4 families activated; active artifacts passed held-out tests |
| 0011 | Does retrieval plus an executable artifact reduce reuse cost? | Artifact reuse: 8/8 with zero model calls |
| 0012 | What fails as documents and learned state scale? | Storage/indexing scaled; generic acquisition degraded sharply |
| 0013 | Can faceted retrieval and scoped composition control scale? | 100k metadata retrieval remained top-1/top-5 correct |
| 0014 | Can the frozen model rank, decompose, and acquire from context? | Context reduction about 90–95%; decomposition 1/3; new activation 0/3 |
| 0015 | Can structured synthesis and diagnostic repair fix acquisition? | Retrieval 5/5, but activation 0/5 in every arm |
| 0016 | Is contract induction the main bottleneck? | Complete contracts: 0/8; semantic invariants remained unresolved |
| 0017 | Can the model produce minimal executable semantic IR? | Direct IR generation: 0/8; model-free oracle compiler: 8/8 |
| 0018 | Can search generate candidates while the model only ranks them? | Correct candidate coverage: 12/12; model ranking: 9/12; random: 3/12 |
| 0019 | Can canonicalization remove ambiguity, and does the model use documentation semantics? | Canonicalized acquisition: 20/20; correct-doc ranking: 6/20; random: 8/20 |
| 0020 | Do verified procedures accumulate, persist, transfer, and compose? | 32/32 acquired; cold-restart reuse 100%; final transfer 48/48 |

## Performance and engineering results

### Model-side learning boundary

When the model was asked to generate new semantic programs, performance collapsed:

| Capability | Result |
|---|---:|
| Novel acquisition at Experiment 0014 | 0/3 |
| Structured/full/repair acquisition at Experiment 0015 | 0/5 |
| Complete contracts at Experiment 0016 | 0/8 |
| Documentation-to-IR at Experiment 0017 | 0/8 |
| Documentation-based candidate ranking at Experiment 0019 | 6/20 |
| No-documentation ranking at Experiment 0019 | 7/20 |
| Fixed random ranking baseline | 8/20 |
| Counterfactual documentation following | 0/8 |

The model was useful as a bounded assistant in some conditions, but it was not a reliable general semantic learner in this study.

### Deterministic acquisition and verification

The deterministic side improved consistently:

| Capability | Result |
|---|---:|
| Symbolic hypothesis search, Experiment 0002 | 8/8 |
| Typed composition, Experiment 0006 | 9/9 |
| Indexed retrieval at 100k metadata entries, Experiment 0013 | 100% top-1/top-5 |
| Correct candidate generation, Experiment 0018 | 12/12 |
| Search plus canonicalization, Experiment 0019 | 20/20 ACTIVE |
| Sequential acquisition, Experiment 0020 | 32/32 ACTIVE |
| Cold-restart reuse, Experiment 0020 | 100% |
| Final direct/near/compositional transfer, Experiment 0020 | 48/48 |

### Efficiency

The study measured external state growth separately from active state:

- Stored learned state at the 100k fixture: approximately 51.57 MB.
- Active learned state per query: approximately 415 B in the measured configuration.
- Context compression in Experiment 0014: approximately 90–95%.
- Experiment 0019 hybrid acquisition: 20/20 ACTIVE with zero model calls.
- Experiment 0020 learned hot path: 48/48 with zero incremental model calls.
- Experiment 0020 measured approximately 25 µs p50 and 67 µs p95 on the learned hot path, compared with 1.165 s p50 and 1.568 s p95 for the paired vanilla model condition.

The hot-path speedup is procedural execution and model bypass; it is not evidence that the model became faster or more capable.

## Final Experiment 0020

Experiment 0020 tested whether verified procedures accumulate as a persistent library rather than functioning as an answer cache.

- 32/32 base procedures became active.
- Wrong activation: 0.
- Six duplicate requests reused existing artifacts.
- Four controlled conflicts remained separate.
- Cold restart preserved 32/32 top-1 retrieval and 100% held-out accuracy.
- Retention remained 100% at 4, 8, 16, 24, and 32-skill checkpoints.
- Final clean transfer:
  - direct: 18/18;
  - near transfer: 12/12;
  - two-procedure composition: 12/12;
  - three-procedure composition: 6/6.
- Learned state stored canonical procedures and metadata, not the final answers for the evaluation tasks.

This is strong evidence for bounded persistent procedural memory. It is not evidence that the frozen language model acquired general cognitive ability.

## Why planned evaluation was not run

> **Planned benchmark and later experiments were not executed because the project’s central hypothesis had already diverged from the implemented architecture. Further evaluation would primarily measure external procedural memory rather than continual cognitive learning in the base model.**

Additional runs inside the same frozen grammar would mainly increase evidence for procedural storage, retrieval, and execution. They would not resolve the original question about general capability growth in the base model.

## Final research interpretation

```text
external procedural capability growth   = demonstrated, bounded
base-model cognitive capability growth = not demonstrated
```

The most accurate description of the completed work is:

> **A verified external procedural learning engine and executable procedural-memory prototype around a frozen small language model.**

The study ended at a useful boundary. It demonstrates a reliable engineered subsystem while falsifying the stronger claim that external procedural state, by itself, turns a small frozen model into a generally improving learner.

## Validity limits

The conclusions are limited by:

- synthetic opaque task families;
- a fixed typed semantic grammar;
- one frozen small-model runtime;
- bounded candidate search;
- finite public, hidden, edge, and held-out sets;
- limited independent seeds;
- no open-world primitive invention;
- no model-weight update, adapter training, or neural crystallization.

The results should therefore be read as a transparent boundary study, not as a claim of general intelligence or broad continual learning.

## Reproduction

The repository includes the source code, deterministic tests, experiment reports, and containerized runtime configuration.

Start the runtime:

```bash
docker compose up --build -d
```

Run diagnostics:

```bash
docker compose exec air-core python -m air_core.cli doctor
docker compose exec air-core python -m air_core.cli smoke
```

Run the deterministic test suite:

```bash
docker compose exec air-core python -m unittest discover -s tests -p 'test_*.py'
```

Run an experiment:

```bash
docker compose exec air-core python -m air_core.cli experiment-020
```

Each report under `docs/experiments/` documents its frozen protocol, command, metrics, limitations, and runtime-output convention. Generated runtime data is intentionally excluded from version control.

## Safety and isolation

The runtime is designed around:

- no credential or host-filesystem dependency;
- no Docker-socket dependency;
- dropped Linux capabilities;
- `no-new-privileges`;
- an unprivileged project process;
- a read-only model-runtime root filesystem;
- static candidate safety checks;
- subprocess isolation and timeouts;
- hidden and edge validation before activation;
- explicit rejection of unsafe, wrong, ambiguous, or unverifiable artifacts.

## Naming and compatibility

The public research identity and repository name are **Verified Procedural Learning Around a Frozen Small Language Model**. Internal runtime identifiers such as `air_core`, `air-core`, `airctl`, and `AIR_*` are retained solely for executable compatibility with the experiment code and saved protocols. They are not personal or host-specific identifiers.

## Repository structure

- `src/air_core/` — experiment runners, storage, retrieval, synthesis, verification, and execution;
- `tests/` — deterministic unit and integration coverage;
- `docs/experiments/` — the complete 0001–0020 experiment record;
- `compose.yaml` — reproducible service topology;
- `.env.example` — non-secret runtime defaults;
- `docker/` — runtime image definition.

The repository contains no personal profile, personal workspace path, host installation detail, credential, or private knowledge-base data.
