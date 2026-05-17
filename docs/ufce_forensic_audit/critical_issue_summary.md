# Critical Issue Summary

Only Critical and Major findings are listed below.

| ID | Severity | Headline | Primary Impact |
| --- | --- | --- | --- |
| UF-AUD-01 | Critical | `Single_F` writes the midpoint index instead of the candidate feature value. | Breaks the published binary-search-style perturbation logic and corrupts search geometry. |
| UF-AUD-05 | Critical | `Double_F` can return non-flipping candidates. | Invalid rows can enter the returned UFCE2 counterfactual set. |
| UF-AUD-06 | Critical | `Triple_F` repeats the same validity failure and omits search-time plausibility. | UFCE3 can emit invalid or implausible rows as candidate counterfactuals. |
| UF-AUD-07 | Critical | Plausibility is enforced post hoc rather than during search. | The active generator is looser than the paper's valid-and-plausible contract. |
| UF-AUD-08 | Critical | Best-counterfactual selection ignores the paper's mixed-distance rule. | Returned "nearest" counterfactuals are not the paper's `z*`. |
| UF-AUD-12 | Critical | Actionability and feasibility metrics use different thresholds and semantics from the paper. | Reported evaluation tables can materially diverge from the paper's metric definitions. |
| UF-AUD-02 | Major | Categorical single-feature reversal is not implemented. | Binary categorical perturbation is asymmetric and incomplete. |
| UF-AUD-03 | Major | User feedback intervals collapse into one-sided additive caps. | The feasible subspace differs from the paper's lower/upper interval contract. |
| UF-AUD-04 | Major | KDTree uses a fixed raw-space radius with no fallback. | Neighborhood geometry and candidate inclusion become dataset-scale dependent. |
| UF-AUD-09 | Major | Distance helpers do not implement the paper's published metric geometry. | Numerical and mixed proximity values are not equation-equivalent to the paper. |
| UF-AUD-10 | Major | Joint proximity is omitted from the observable reporting path. | The active repository snapshot does not operationalize the paper's `delta` metric. |
| UF-AUD-11 | Major | Default presets provide no protected features. | Immutable-feature guarantees are absent from default runs. |
| UF-AUD-13 | Major | DiCE, DiCE-UF, and AR receive constraints asymmetrically. | Baseline fairness is weakened by mixed native vs post-hoc enforcement. |
| UF-AUD-14 | Major | The visible experiment script is narrower than the paper's reported scope. | The snapshot is not, by default, a five-dataset evaluation pipeline. |
| UF-AUD-15 | Major | CV protocol in code is 10-fold, while the paper states fivefold. | Repository CV values cannot be treated as direct implementations of Table 3. |
