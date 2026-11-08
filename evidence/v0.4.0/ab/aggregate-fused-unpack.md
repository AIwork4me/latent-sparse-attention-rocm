# v0.4.0 A/B aggregation (3 receipts, candidates: ['fused_unpack'])

Speedup = interleaved reference p50 / candidate p50, same run. OBSERVED RANGES, not CIs.

| dtype | ctx | phase | stage | cand | pooled med | seed medians (min-max) | all runs (min-max) | ref→cand p50 ms | launches |
|---|---:|---|---|---|---:|---|---|---|---|
| bf16 | 512 | decode_b1 | select | fused_unpack | 1.099x | 1.075-1.103x | 1.075-1.103x | 0.175 → 0.159 | 34→21 |
| bf16 | 512 | decode_b1 | select_read | fused_unpack | 1.082x | 1.081-1.134x | 1.081-1.134x | 0.261 → 0.239 | 37→24 |
| bf16 | 512 | decode_ragged | select | fused_unpack | 1.128x | 1.101-1.134x | 1.101-1.134x | 0.188 → 0.167 | 34→21 |
| bf16 | 512 | decode_ragged | select_read | fused_unpack | 1.033x | 1.029-1.067x | 1.029-1.067x | 0.447 → 0.434 | 37→24 |
| bf16 | 512 | prefill | select | fused_unpack | 1.093x | 1.060-1.152x | 1.060-1.152x | 0.214 → 0.197 | 46→21 |
| bf16 | 512 | prefill | select_read | fused_unpack | 1.025x | 0.999-1.028x | 0.999-1.028x | 2.633 → 2.573 | 37→24 |
| bf16 | 2048 | decode_b1 | select | fused_unpack | 1.150x | 1.123-1.204x | 1.123-1.204x | 0.188 → 0.168 | 34→21 |
| bf16 | 2048 | decode_b1 | select_read | fused_unpack | 1.107x | 1.046-1.164x | 1.046-1.164x | 0.286 → 0.259 | 37→24 |
| bf16 | 2048 | decode_ragged | select | fused_unpack | 1.121x | 1.083-1.229x | 1.083-1.229x | 0.195 → 0.171 | 34→21 |
| bf16 | 2048 | decode_ragged | select_read | fused_unpack | 1.068x | 1.068-1.101x | 1.068-1.101x | 0.504 → 0.472 | 37→24 |
| bf16 | 2048 | prefill | select | fused_unpack | 1.094x | 1.068-1.100x | 1.068-1.100x | 0.982 → 0.898 | 34→21 |
| bf16 | 2048 | prefill | select_read | fused_unpack | 1.010x | 1.010-1.013x | 1.010-1.013x | 11.118 → 11.012 | 37→24 |
| bf16 | 8192 | decode_b1 | select | fused_unpack | 1.104x | 1.084-1.129x | 1.084-1.129x | 0.185 → 0.165 | 34→21 |
| bf16 | 8192 | decode_b1 | select_read | fused_unpack | 1.089x | 1.079-1.195x | 1.079-1.195x | 0.288 → 0.252 | 37→24 |
| bf16 | 8192 | decode_ragged | select | fused_unpack | 1.026x | 1.018-1.060x | 1.018-1.060x | 0.219 → 0.213 | 34→21 |
| bf16 | 8192 | decode_ragged | select_read | fused_unpack | 1.034x | 1.033-1.058x | 1.033-1.058x | 0.512 → 0.494 | 37→24 |
| bf16 | 8192 | prefill | select | fused_unpack | 1.267x | 1.267-1.276x | 1.267-1.276x | 12.835 → 10.066 | 52→39 |
| bf16 | 8192 | prefill | select_read | fused_unpack | 1.032x | 1.031-1.035x | 1.031-1.035x | 60.993 → 59.155 | 55→42 |
| bf16 | 16384 | decode_b1 | select | fused_unpack | 1.214x | 1.177-1.259x | 1.177-1.259x | 0.188 → 0.154 | 34→21 |
| bf16 | 16384 | decode_b1 | select_read | fused_unpack | 1.097x | 1.097-1.132x | 1.097-1.132x | 0.263 → 0.240 | 37→24 |
| bf16 | 16384 | decode_ragged | select | fused_unpack | 1.076x | 1.057-1.085x | 1.057-1.085x | 0.203 → 0.187 | 34→21 |
| bf16 | 16384 | decode_ragged | select_read | fused_unpack | 1.052x | 1.019-1.090x | 1.019-1.090x | 0.514 → 0.493 | 37→24 |
| bf16 | 16384 | prefill | select | fused_unpack | 1.184x | 1.173-1.188x | 1.173-1.188x | 32.788 → 27.640 | 52→39 |
| bf16 | 16384 | prefill | select_read | fused_unpack | 1.036x | 1.035-1.036x | 1.035-1.036x | 145.210 → 140.264 | 55→42 |
| bf16 | 32768 | decode_b1 | select | fused_unpack | 1.196x | 1.193-1.201x | 1.193-1.201x | 0.497 → 0.417 | 34→21 |
| bf16 | 32768 | decode_b1 | select_read | fused_unpack | 1.125x | 1.105-1.135x | 1.105-1.135x | 0.268 → 0.242 | 37→24 |
| bf16 | 32768 | decode_ragged | select | fused_unpack | 1.050x | 1.043-1.075x | 1.043-1.075x | 0.210 → 0.200 | 34→21 |
| bf16 | 32768 | decode_ragged | select_read | fused_unpack | 1.077x | 1.048-1.095x | 1.048-1.095x | 0.565 → 0.516 | 37→24 |
| bf16 | 32768 | prefill | select | fused_unpack | 1.115x | 1.113-1.118x | 1.113-1.118x | 97.601 → 87.502 | 52→39 |
| bf16 | 32768 | prefill | select_read | fused_unpack | 1.027x | 1.026-1.028x | 1.026-1.028x | 351.441 → 342.422 | 55→42 |
| fp16 | 512 | decode_b1 | select | fused_unpack | 1.170x | 1.143-1.173x | 1.143-1.173x | 0.181 → 0.154 | 34→21 |
| fp16 | 512 | decode_b1 | select_read | fused_unpack | 1.145x | 1.115-1.180x | 1.115-1.180x | 0.248 → 0.214 | 37→24 |
| fp16 | 512 | decode_ragged | select | fused_unpack | 1.177x | 1.129-1.202x | 1.129-1.202x | 0.177 → 0.157 | 34→21 |
| fp16 | 512 | decode_ragged | select_read | fused_unpack | 1.056x | 1.041-1.059x | 1.041-1.059x | 0.411 → 0.388 | 37→24 |
| fp16 | 512 | prefill | select | fused_unpack | 1.184x | 1.134-1.230x | 1.134-1.230x | 0.193 → 0.163 | 34→21 |
| fp16 | 512 | prefill | select_read | fused_unpack | 1.026x | 1.020-1.032x | 1.020-1.032x | 2.141 → 2.087 | 37→24 |
| fp16 | 2048 | decode_b1 | select | fused_unpack | 1.158x | 1.095-1.165x | 1.095-1.165x | 0.177 → 0.153 | 34→21 |
| fp16 | 2048 | decode_b1 | select_read | fused_unpack | 1.116x | 1.086-1.118x | 1.086-1.118x | 0.266 → 0.238 | 37→24 |
| fp16 | 2048 | decode_ragged | select | fused_unpack | 1.148x | 1.128-1.151x | 1.128-1.151x | 0.199 → 0.173 | 34→21 |
| fp16 | 2048 | decode_ragged | select_read | fused_unpack | 1.049x | 1.049-1.057x | 1.049-1.057x | 0.399 → 0.381 | 37→24 |
| fp16 | 2048 | prefill | select | fused_unpack | 1.104x | 1.097-1.120x | 1.097-1.120x | 0.988 → 0.881 | 34→21 |
| fp16 | 2048 | prefill | select_read | fused_unpack | 1.013x | 1.010-1.013x | 1.010-1.013x | 9.469 → 9.346 | 37→24 |
| fp16 | 8192 | decode_b1 | select | fused_unpack | 1.186x | 1.168-1.189x | 1.168-1.189x | 0.184 → 0.157 | 34→21 |
| fp16 | 8192 | decode_b1 | select_read | fused_unpack | 1.088x | 1.085-1.109x | 1.085-1.109x | 0.289 → 0.266 | 37→24 |
| fp16 | 8192 | decode_ragged | select | fused_unpack | 1.039x | 1.027-1.054x | 1.027-1.054x | 0.222 → 0.213 | 34→21 |
| fp16 | 8192 | decode_ragged | select_read | fused_unpack | 1.052x | 1.039-1.057x | 1.039-1.057x | 0.470 → 0.445 | 37→24 |
| fp16 | 8192 | prefill | select | fused_unpack | 1.256x | 1.251-1.269x | 1.251-1.269x | 12.546 → 9.990 | 52→39 |
| fp16 | 8192 | prefill | select_read | fused_unpack | 1.035x | 1.030-1.038x | 1.030-1.038x | 57.465 → 55.734 | 55→42 |
| fp16 | 16384 | decode_b1 | select | fused_unpack | 1.149x | 1.117-1.182x | 1.117-1.182x | 0.204 → 0.183 | 34→21 |
| fp16 | 16384 | decode_b1 | select_read | fused_unpack | 1.139x | 1.071-1.153x | 1.071-1.153x | 0.267 → 0.239 | 37→24 |
| fp16 | 16384 | decode_ragged | select | fused_unpack | 1.062x | 1.056-1.082x | 1.056-1.082x | 0.203 → 0.192 | 34→21 |
| fp16 | 16384 | decode_ragged | select_read | fused_unpack | 1.050x | 1.046-1.079x | 1.046-1.079x | 0.469 → 0.448 | 37→24 |
| fp16 | 16384 | prefill | select | fused_unpack | 1.177x | 1.172-1.184x | 1.172-1.184x | 33.052 → 28.077 | 52→39 |
| fp16 | 16384 | prefill | select_read | fused_unpack | 1.036x | 1.035-1.036x | 1.035-1.036x | 143.717 → 138.841 | 55→42 |
| fp16 | 32768 | decode_b1 | select | fused_unpack | 1.219x | 1.181-1.222x | 1.181-1.222x | 0.509 → 0.420 | 34→21 |
| fp16 | 32768 | decode_b1 | select_read | fused_unpack | 1.062x | 1.039-1.067x | 1.039-1.067x | 0.265 → 0.249 | 37→24 |
| fp16 | 32768 | decode_ragged | select | fused_unpack | 1.060x | 1.048-1.065x | 1.048-1.065x | 0.235 → 0.222 | 34→21 |
| fp16 | 32768 | decode_ragged | select_read | fused_unpack | 1.069x | 1.039-1.096x | 1.039-1.096x | 0.547 → 0.499 | 37→24 |
| fp16 | 32768 | prefill | select | fused_unpack | 1.113x | 1.112-1.114x | 1.112-1.114x | 97.261 → 87.402 | 52→39 |
| fp16 | 32768 | prefill | select_read | fused_unpack | 1.026x | 1.024-1.026x | 1.024-1.026x | 357.652 → 348.715 | 55→42 |
