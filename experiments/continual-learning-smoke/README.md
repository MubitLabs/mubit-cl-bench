# Continual-learning smoke run

An n=8 directional smoke run of Mubit on the continual-learning-bench database-exploration task (Gemini 3 Flash backbone).
Result as written in MUBIT_EXPERIMENT.md: Mubit gain -0.467 (stateful 0.083 vs stateless 0.142, $0.30, 69 LLM calls); ICL baseline gain +0.133 (0.117 vs 0.100, $0.055, 31 calls).
The guide says n=8 / 1 rollout is not statistically meaningful and only validates wiring and direction.
The probe scripts in scripts/ check Mubit retrieval isolation and recall; they read MUBIT_API_KEY from the environment.
Upstream benchmark: github.com/pgasawa/continual-learning-bench (the Mubit system code itself is not included here).
