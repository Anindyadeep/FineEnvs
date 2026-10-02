# RetroEnv v2 eval board

Split `eval`, 150 tasks, 1 attempt(s), 16 model turns, toolset `full`, tool budget 32. Every tool call and reward came from the OpenEnv server. Pass needs every required route verified against a hidden patent route; exact route is the share of tasks where a submitted route matches one step for step.

| Model | Tasks | Pass@1 (95% CI) | Exact route | Reward | Steps | Stock | Graph | Tool calls | No emit | Refused | Cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-opus-5-5 | 150/150 | 0.560 [0.48, 0.64] | 0.593 | 0.750 | 0.727 | 0.793 | 0.831 | 9.8 | 0.160 | 0.160 | $13.94 |
| claude-sonnet-5-5 | 150/150 | 0.300 [0.23, 0.38] | 0.340 | 0.626 | 0.499 | 0.814 | 0.835 | 10.9 | 0.000 | 0.000 | $9.76 |
| gpt-5.6-sol | 150/150 | 0.280 [0.21, 0.36] | 0.320 | 0.705 | 0.554 | 0.914 | 0.989 | 14.9 | 0.000 | 0.000 | $23.25 |
| deepseek-v4.1-flash | 150/150 | 0.107 [0.07, 0.17] | 0.127 | 0.436 | 0.248 | 0.601 | 0.678 | 11.2 | 0.313 | 0.000 | $5.03 |
| gpt-5.6-luna | 150/150 | 0.073 [0.04, 0.13] | 0.100 | 0.565 | 0.292 | 0.823 | 0.956 | 14.8 | 0.000 | 0.000 | $2.29 |
| qwen3.8-27b | 150/150 | 0.013 [0.00, 0.05] | 0.007 | 0.138 | 0.033 | 0.187 | 0.215 | 6.6 | 0.693 | 0.000 | $8.27 |

### By required routes (exact route rate)

| Model | single_route | two_route |
|---|---|---|
| claude-opus-5-5 | 0.583 (n=132) | 0.667 (n=18) |
| claude-sonnet-5-5 | 0.318 (n=132) | 0.500 (n=18) |
| gpt-5.6-sol | 0.311 (n=132) | 0.389 (n=18) |
| deepseek-v4.1-flash | 0.121 (n=132) | 0.167 (n=18) |
| gpt-5.6-luna | 0.099 (n=132) | 0.111 (n=18) |
| qwen3.8-27b | 0.008 (n=132) | 0.000 (n=18) |

### By step cap (exact route rate)

| Model | 2 | 3 |
|---|---|---|
| claude-opus-5-5 | 0.662 (n=80) | 0.514 (n=70) |
| claude-sonnet-5-5 | 0.438 (n=80) | 0.229 (n=70) |
| gpt-5.6-sol | 0.450 (n=80) | 0.171 (n=70) |
| deepseek-v4.1-flash | 0.200 (n=80) | 0.043 (n=70) |
| gpt-5.6-luna | 0.175 (n=80) | 0.014 (n=70) |
| qwen3.8-27b | 0.013 (n=80) | 0.000 (n=70) |

### By heuristic tier (exact route rate)

| Model | easy | hard | medium |
|---|---|---|---|
| claude-opus-5-5 | 0.675 (n=83) | 0.391 (n=23) | 0.545 (n=44) |
| claude-sonnet-5-5 | 0.373 (n=83) | 0.217 (n=23) | 0.341 (n=44) |
| gpt-5.6-sol | 0.434 (n=83) | 0.130 (n=23) | 0.204 (n=44) |
| deepseek-v4.1-flash | 0.169 (n=83) | 0.000 (n=23) | 0.114 (n=44) |
| gpt-5.6-luna | 0.108 (n=83) | 0.087 (n=23) | 0.091 (n=44) |
| qwen3.8-27b | 0.000 (n=83) | 0.000 (n=23) | 0.023 (n=44) |

## Reading the table

**No emit** is the share of episodes the model never closed with `emit_routes`; the harness then submits an empty route set, which scores the 0.05 floor. **Refused** is the share the provider declined on safety grounds, which counts as a failed episode and is never re-routed to another model. Where the two columns match, every unclosed episode was a refusal, and the model's reward is held down by requests it did not answer rather than by its chemistry.

**Steps**, **Stock** and **Graph** are the step-correctness, stock-correctness and graph-validity reward components. A model can score well on stock and graph while failing the task: those measure that the submitted tree is well formed and its leaf claims are truthful, not that the route matches the patent.
