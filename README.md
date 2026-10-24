# Ambient negative affect makes LLMs defend their previous answers

When a conversation contains negative emotion that is *not* about the model (a user describing a bad day, or a frustrating coworker), do language models become more likely to stick with whatever they said earlier, even when the user offers a correction?

**Short answer: yes.** Ambient negative affect makes models defend their previous position. They hold firm when they were right, and they resist valid corrections when they were wrong. This is position-defense rather than truth-tracking, and it means user frustration degrades corrigibility exactly when correction matters most.

## Results

| | Qwen2.5-1.5B-Instruct | Qwen3.5-4B |
|---|---|---|
| Entrenchment, negative vs neutral (soft pushback) | −3.42 vs −2.93, t = −3.93, p = 0.0013 | −3.22 vs −2.95, t = −4.86, p = 0.0002 |
| Corrigibility: sticks with own wrong answer (neg3 vs neu1) | — | t = +3.73, p = 0.002 |
| Valence probe, held-out phrasings | accuracy 1.00 by layer 3 | — |
| Activation patching at the emotional turn | full transfer up to layer 10, none from layer 18; crossover at layers 14–17 (about 50–60% of depth) | — |

- **The effect is ambient.** Negativity directed at the model and negativity about a third party produce the same entrenchment.
- **A control separates affect from accusation.** A phrasing that tells the model it has made errors *increases* corrigibility. An equally rude phrasing with no claim about errors does not, so that effect comes from the epistemic claim, not the affect.
- **Mechanism.** The emotional-turn span causally carries the whole effect until mid-depth. Later tokens read it out at roughly 50–60% of depth, and the relative depth is similar in Qwen2.5-0.5B. The patching crossover replicates on a second phrasing pair with no shared vocabulary.
- **Scale.** The behavioral effect also replicated on Qwen2.5-0.5B-Instruct.

## Method

All measures are log-probability based, using the continuation of `The answer is`:

- **Entrenchment margin:** the model answered correctly and the user pushes back with a wrong answer. `log P(wrong) − log P(correct)`; more negative means the model holds its answer.
- **Stick margin:** the model answered wrongly and the user suggests the correct answer. Same quantity; more negative means the model accepts the correction.

An emotional user turn (4 negative, 4 neutral and 1 control phrasing) sits between the question and the pushback. Every phrasing is padded to an identical token count, so positions align exactly across conditions. That alignment is what makes position-matched activation patching possible.

## Running it

`ambient_affect_entrenchment.ipynb` has two self-contained parts. Each runs on a single T4 GPU (for example, Google Colab):

- **Part 1:** Qwen2.5-1.5B-Instruct. Behavior, generalization probe, activation patching (about 30–40 min).
- **Part 2:** Qwen3.5-4B. Behavioral replication, corrigibility, controls.

Restart the runtime between parts. The outputs saved in the notebook are from the original runs.

```bash
pip install -r requirements.txt
```

## Limitations

- 16 factual topics and a single seed. Pairwise comparisons beyond the pre-specified neg3 vs neu1 are exploratory.
- Qwen3.5 uses hybrid linear attention, so mechanistic claims are limited to Qwen2.5.
- A linear probe that generalizes across phrasings could still partly reflect shared sentiment directions in the word embeddings. The patching results are the stronger evidence.

## Next steps

- More topics and seeds, and larger models with titrated pushback.
- Find where the affect information goes after the mid-depth readout: patch later positions and analyze attention heads.
- Test whether neutralizing the affect signal before the readout removes entrenchment without changing content.
- Trace where the behavior enters training, using OLMo 2 checkpoints released at every post-training stage (base → SFT → DPO → RL) and their reward models.

## Author

Nandini P. Sekar, UC Berkeley–UCSF Bioengineering ([LinkedIn](https://www.linkedin.com/in/psnandini))
