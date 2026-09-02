# Related Work

This study sits at the intersection of continual learning, program synthesis, and external-memory language-agent systems. The comparison below is intentionally narrow: it identifies the closest conceptual neighbors and clarifies what the present experiments do and do not claim.

## Continual learning and forgetting

Kirkpatrick et al. study catastrophic forgetting and parameter-level consolidation in neural networks:

- [Overcoming catastrophic forgetting in neural networks](https://doi.org/10.1073/pnas.1611835114)

Lopez-Paz and Ranzato introduce episodic-memory constraints for continual learning:

- [Gradient Episodic Memory for Continual Learning](https://openreview.net/forum?id=Hrk2E8Ytv)

The present study differs from both lines of work in a decisive way: the base language-model parameters remain frozen. No gradient update, adapter, replay buffer for parameter training, or weight consolidation is used. Therefore, the positive result here is external procedural capability growth, not parameter-level continual learning. The negative result concerns whether the frozen model can independently acquire new semantic contracts from documentation and verified examples.

## Program synthesis and executable abstractions

DreamCoder demonstrates how program-learning systems can accumulate reusable abstractions through a wake-sleep process:

- [DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning](https://arxiv.org/abs/2006.08381)

This study shares the emphasis on executable abstractions and reuse, but uses a deliberately bounded typed grammar, explicit validation gates, behavioral canonicalization, and persistent artifact storage. It does not claim open-ended library invention or unrestricted program synthesis.

## External memory for language agents

MemGPT frames long-term context management as an operating-system-like memory problem for language agents:

- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)

Generative Agents combine memory retrieval, reflection, and planning to support believable interactive behavior:

- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)

The present system is narrower and more executable. It stores verified procedures, contracts, provenance, and integrity metadata rather than primarily storing conversations, summaries, or autobiographical memories. The evaluation focuses on activation correctness, rejection of unsafe or ambiguous artifacts, persistence across restart, retrieval, composition, and transfer.

## Position of this study

The contribution is a boundary result:

1. External procedural state can become a reliable, persistent, model-free execution layer when candidate generation and verification are bounded.
2. That layer should not be conflated with the frozen model learning new general semantics.
3. A negative result on model-side semantic acquisition can coexist with a positive result for the surrounding verified system.

The experiment reports in this repository provide the protocol and numerical evidence for these claims. They are not intended as a comprehensive survey or as a claim that this architecture subsumes the cited approaches.
