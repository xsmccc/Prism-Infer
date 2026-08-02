# Repeated visual-context quality

| Dataset / cohort | Configuration | Cross-question Prefix reuse | Samples with token deletion | Dropped visual tokens | Result |
|---|---|---:|---:|---:|---:|
| MuirBench, all multi-question samples | Dense official interleaved | No | 0 | 0 | 49/85 (57.65%) |
| MuirBench, all multi-question samples | Dense labeled media-first | Yes | 0 | 0 | 46/85 (54.12%) |
| MuirBench, paired actual-deletion cohort | Dense labeled media-first | Yes | 0 | 0 | 27/49 (55.10%) |
| MuirBench, paired actual-deletion cohort | Attention Top-k per question | No | 49 | 52,120 | 20/49 (40.82%) |
| MuirBench, paired actual-deletion cohort | First-question Attention Top-k reused | Yes | 49 | 52,120 | 20/49 (40.82%) |
| MuirBench, paired actual-deletion cohort | Query-agnostic Uniform reused | Yes | 49 | 52,120 | 20/49 (40.82%) |
| DocVQA, repeated-document set | Dense | Yes | 0 | 0 | ANLS 0.93335 (190 samples) |
| DocVQA, repeated-document set | Uniform | Yes | 0 | 0 | ANLS 0.93335 (190 samples) |
| MVBench, same-video multi-question set | Dense | Yes | 0 | 0 | 183/252 (72.62%) |
| MVBench, same-video multi-question set | Uniform, video deletion explicitly enabled | Yes | 252 | 20,064 | 113/252 (44.84%) |

MuirBench uses the official-compatible multiple-choice score. The paired cohort contains exactly the
49 samples where the compact variants physically removed visual tokens. On that cohort, Uniform
changed 23 outcomes relative to Dense: 15 Dense wins and 8 Uniform wins, for a net loss of seven
correct answers. The equal DocVQA scores do not establish compression quality because its 768-token
image floor prevented deletion in every sample. The MVBench loss is why video token deletion is off
by default.
