---
name: flywire-genome-frame
description: >-
  Design FlyWire under the bio-like NN constructor frame. Use when planning
  demos, architectures, learning, visuals, or "what to build next"; when tempted
  by Shiu LIF, MB conditioning, GNN-on-graph, digital-twin, or MLP-style diagrams.
---

# FlyWire genome frame (workflow)

## Mandatory product frame

- **Goal:** конструктор биологически подобной нейросети.
- **Method:** ЦНС мухи как образец для моделирования и reverse engineering.
- **Requirement:** точное визуальное отображение процессов в этой ЦНС (UI не врёт; не схемы обычных нейросетей).

## Before proposing work

1. State the proposal in one sentence against the product frame above.
2. Ask: could this sentence appear in Shiu 2024, a standard MB learning tutorial, or "GNN on connectome" paper? If yes → rewrite or kill.
3. Map to one primary track:
   - **Genome** — rules that generate/approximate wiring
   - **Stimulation policy** — frozen graph, learn how to drive it
   - **Sockets** — module I/O and recomposition

## Default stack for a new idea

| Step | Do | Don't |
|------|----|-------|
| Probe | Minimal dynamics only to test a claim | Full-brain paper replication |
| Plasticity | Rules, ports, or stimulation | All-edge backprop by default |
| Demo | Recipe changed → behavior/competence changed | "We matched known circuit X" |
| Visual | Honest tract / hop / activation / carve | MLP layers, lying overlays |
| Next | Wasp-like / alien assembly from mutated recipe | Bigger fly twin |

## Output shape when advising

Keep answers short. End with: **track** (1/2/3), **forbidden trap avoided**, **smallest next experiment**.
