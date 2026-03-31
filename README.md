# Any House Any Task: Scalable Long-Horizon Planning for Abstract Human Tasks

[![arXiv](https://img.shields.io/badge/arXiv-2602.12244-b31b1b.svg)](https://arxiv.org/abs/2602.12244)
[![Project Page](https://img.shields.io/badge/Project-Website-blue)](xxxxx)

This repository contains the official implementation of **AHAT** (Any House Any Task), a household task planner optimized for scalable, long-horizon planning in large environments given ambiguous human instructions.

## News 📢
- **[2026-02]** Our paper is available on [arXiv](https://arxiv.org/abs/2602.12244)!
- **[2026-02]** Project website is live at [here](xxxxx).
- **[Coming Soon]** We are currently preparing the codebase for open-source release. Stay tuned!

## Abstract
Open world language conditioned task planning is crucial for robots operating in large-scale household environments. While many recent works attempt to address this problem using Large Language Models (LLMs) via prompting or training, a key challenge remains scalability. Performance often degrades rapidly with increasing environment size, plan length, instruction ambiguity, and constraint complexity. In this work, we propose **Any House Any Task (AHAT)**, a household task planner optimized for long-horizon planning in large environments given ambiguous human instructions. At its core, AHAT utilizes an LLM trained to map task instructions and textual scene graphs into grounded subgoals defined in the **Planning Domain Definition Language (PDDL)**. These subgoals are subsequently solved to generate feasible and optimal long-horizon plans through explicit symbolic reasoning. To enhance the model's ability to decompose complex and ambiguous intentions, we introduce **TGPO**, a novel reinforcement learning algorithm that integrates external correction of intermediate reasoning traces into **Group Relative Policy Optimization (GRPO)**. Experiments demonstrate that AHAT achieves significant performance gains over state-of-the-art prompting, planning, and learning methods, particularly in human-style household tasks characterized by brief instructions but requiring complex execution plans.

## Key Features
- **Scalable Long-Horizon Planning:** Robust performance in large-scale household environments without degradation from plan length or constraint complexity.
- **LLM-to-PDDL Grounding:** Maps ambiguous, brief human instructions and textual scene graphs into structured, executable PDDL subgoals.
- **TGPO Reinforcement Learning:** A novel RL algorithm building on GRPO that incorporates external corrections of intermediate reasoning traces to handle complex intentions.
- **Symbolic Reasoning Integration:** Leverages explicit symbolic solvers on generated subgoals to guarantee feasible and optimal execution plans.

## Citation
If you find our work or code useful, please cite our paper:

```bibtex
@article{liu2026ahat,
  title={Any House Any Task: Scalable Long-Horizon Planning for Abstract Human Tasks},
  author={Liu, Zhihong and Li, Yang and Huang, Rengming and Lu, Cewu and Cai, Panpan},
  journal={arXiv preprint arXiv:2602.12244},
  year={2026}
}
