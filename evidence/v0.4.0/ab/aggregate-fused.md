# v0.4.0 A/B aggregation (9 receipts, candidates: ['fused'])

Speedup = interleaved reference p50 / candidate p50, same run. OBSERVED RANGES, not CIs.

| dtype | ctx | phase | stage | cand | pooled med | seed medians (min-max) | all runs (min-max) | ref→cand p50 ms | launches |
|---|---:|---|---|---|---:|---|---|---|---|
| bf16 | 512 | decode_b1 | select | fused | 1.292x | 1.262-1.382x | 1.210-1.425x | 0.186 → 0.143 | 34→16 |
| bf16 | 512 | decode_b1 | select_read | fused | 1.199x | 1.187-1.206x | 1.155-1.263x | 0.256 → 0.213 | 37→19 |
| bf16 | 512 | decode_ragged | select | fused | 1.283x | 1.251-1.318x | 1.186-1.352x | 0.186 → 0.147 | 34→16 |
| bf16 | 512 | decode_ragged | select_read | fused | 1.157x | 1.124-1.168x | 1.097-1.190x | 0.459 → 0.394 | 37→19 |
| bf16 | 512 | prefill | select | fused | 1.272x | 1.247-1.327x | 1.237-1.353x | 0.207 → 0.162 | 46→16 |
| bf16 | 512 | prefill | select_read | fused | 1.047x | 1.033-1.067x | 1.023-1.074x | 2.636 → 2.530 | 37→19 |
| bf16 | 2048 | decode_b1 | select | fused | 1.267x | 1.258-1.387x | 1.236-1.402x | 0.185 → 0.144 | 34→16 |
| bf16 | 2048 | decode_b1 | select_read | fused | 1.200x | 1.146-1.234x | 1.122-1.262x | 0.264 → 0.220 | 37→19 |
| bf16 | 2048 | decode_ragged | select | fused | 1.304x | 1.261-1.361x | 1.248-1.418x | 0.189 → 0.145 | 34→16 |
| bf16 | 2048 | decode_ragged | select_read | fused | 1.154x | 1.147-1.167x | 1.092-1.248x | 0.485 → 0.415 | 37→19 |
| bf16 | 2048 | prefill | select | fused | 2.168x | 2.132-2.171x | 2.099-2.203x | 0.946 → 0.438 | 34→16 |
| bf16 | 2048 | prefill | select_read | fused | 1.048x | 1.045-1.052x | 1.029-1.059x | 10.489 → 10.072 | 37→19 |
| bf16 | 8192 | decode_b1 | select | fused | 1.343x | 1.310-1.363x | 1.289-1.442x | 0.225 → 0.165 | 34→16 |
| bf16 | 8192 | decode_b1 | select_read | fused | 1.176x | 1.153-1.246x | 1.125-1.249x | 0.313 → 0.263 | 37→19 |
| bf16 | 8192 | decode_ragged | select | fused | 1.345x | 1.328-1.435x | 1.296-1.468x | 0.225 → 0.168 | 34→16 |
| bf16 | 8192 | decode_ragged | select_read | fused | 1.164x | 1.137-1.184x | 1.116-1.202x | 0.522 → 0.451 | 37→19 |
| bf16 | 8192 | prefill | select | fused | 3.761x | 3.613-3.873x | 3.608-3.943x | 12.639 → 3.335 | 52→16 |
| bf16 | 8192 | prefill | select_read | fused | 1.129x | 1.127-1.130x | 1.119-1.131x | 60.517 → 53.623 | 55→19 |
| bf16 | 16384 | decode_b1 | select | fused | 1.346x | 1.346-1.349x | 1.332-1.390x | 0.214 → 0.159 | 34→16 |
| bf16 | 16384 | decode_b1 | select_read | fused | 1.211x | 1.203-1.224x | 1.137-1.239x | 0.328 → 0.283 | 37→19 |
| bf16 | 16384 | decode_ragged | select | fused | 1.333x | 1.310-1.374x | 1.303-1.461x | 0.230 → 0.173 | 34→16 |
| bf16 | 16384 | decode_ragged | select_read | fused | 1.144x | 1.129-1.169x | 1.091-1.205x | 0.517 → 0.456 | 37→19 |
| bf16 | 16384 | prefill | select | fused | 3.545x | 3.467-3.570x | 3.424-3.606x | 32.981 → 9.287 | 52→16 |
| bf16 | 16384 | prefill | select_read | fused | 1.178x | 1.177-1.179x | 1.176-1.181x | 145.923 → 123.730 | 55→19 |
| bf16 | 32768 | decode_b1 | select | fused | 1.384x | 1.373-1.389x | 1.364-1.432x | 0.512 → 0.369 | 34→16 |
| bf16 | 32768 | decode_b1 | select_read | fused | 1.199x | 1.192-1.238x | 1.173-1.242x | 0.317 → 0.266 | 37→19 |
| bf16 | 32768 | decode_ragged | select | fused | 1.344x | 1.324-1.373x | 1.323-1.438x | 0.229 → 0.166 | 34→16 |
| bf16 | 32768 | decode_ragged | select_read | fused | 1.197x | 1.177-1.214x | 1.127-1.216x | 0.570 → 0.484 | 37→19 |
| bf16 | 32768 | prefill | select | fused | 2.962x | 2.952-2.969x | 2.946-2.972x | 98.048 → 33.118 | 52→16 |
| bf16 | 32768 | prefill | select_read | fused | 1.226x | 1.225-1.228x | 1.219-1.229x | 349.361 → 284.900 | 55→19 |
| fp16 | 512 | decode_b1 | select | fused | 1.304x | 1.299-1.337x | 1.274-1.403x | 0.221 → 0.170 | 34→16 |
| fp16 | 512 | decode_b1 | select_read | fused | 1.258x | 1.208-1.265x | 1.175-1.286x | 0.324 → 0.258 | 37→19 |
| fp16 | 512 | decode_ragged | select | fused | 1.385x | 1.293-1.393x | 1.246-1.419x | 0.217 → 0.168 | 34→16 |
| fp16 | 512 | decode_ragged | select_read | fused | 1.147x | 1.100-1.184x | 1.075-1.203x | 0.440 → 0.391 | 37→19 |
| fp16 | 512 | prefill | select | fused | 1.246x | 1.236-1.253x | 1.230-1.326x | 0.241 → 0.186 | 34→16 |
| fp16 | 512 | prefill | select_read | fused | 1.057x | 1.051-1.068x | 1.034-1.075x | 2.193 → 2.079 | 37→19 |
| fp16 | 2048 | decode_b1 | select | fused | 1.322x | 1.303-1.329x | 1.261-1.385x | 0.227 → 0.171 | 34→16 |
| fp16 | 2048 | decode_b1 | select_read | fused | 1.180x | 1.155-1.206x | 1.152-1.233x | 0.308 → 0.255 | 37→19 |
| fp16 | 2048 | decode_ragged | select | fused | 1.328x | 1.321-1.342x | 1.204-1.378x | 0.219 → 0.171 | 34→16 |
| fp16 | 2048 | decode_ragged | select_read | fused | 1.159x | 1.152-1.160x | 1.117-1.204x | 0.432 → 0.373 | 37→19 |
| fp16 | 2048 | prefill | select | fused | 2.131x | 2.095-2.139x | 2.074-2.197x | 0.923 → 0.433 | 34→16 |
| fp16 | 2048 | prefill | select_read | fused | 1.059x | 1.051-1.062x | 1.038-1.067x | 9.060 → 8.617 | 37→19 |
| fp16 | 8192 | decode_b1 | select | fused | 1.361x | 1.340-1.368x | 1.321-1.420x | 0.235 → 0.173 | 34→16 |
| fp16 | 8192 | decode_b1 | select_read | fused | 1.212x | 1.201-1.218x | 1.166-1.276x | 0.331 → 0.281 | 37→19 |
| fp16 | 8192 | decode_ragged | select | fused | 1.347x | 1.343-1.347x | 1.341-1.396x | 0.236 → 0.176 | 34→16 |
| fp16 | 8192 | decode_ragged | select_read | fused | 1.173x | 1.170-1.210x | 1.085-1.236x | 0.459 → 0.389 | 37→19 |
| fp16 | 8192 | prefill | select | fused | 3.851x | 3.820-3.897x | 3.746-3.943x | 12.651 → 3.282 | 52→16 |
| fp16 | 8192 | prefill | select_read | fused | 1.135x | 1.134-1.139x | 1.132-1.139x | 57.279 → 50.574 | 55→19 |
| fp16 | 16384 | decode_b1 | select | fused | 1.377x | 1.336-1.400x | 1.282-1.413x | 0.226 → 0.165 | 34→16 |
| fp16 | 16384 | decode_b1 | select_read | fused | 1.202x | 1.194-1.208x | 1.178-1.230x | 0.306 → 0.255 | 37→19 |
| fp16 | 16384 | decode_ragged | select | fused | 1.343x | 1.318-1.347x | 1.305-1.373x | 0.225 → 0.166 | 34→16 |
| fp16 | 16384 | decode_ragged | select_read | fused | 1.205x | 1.200-1.225x | 1.158-1.250x | 0.495 → 0.412 | 37→19 |
| fp16 | 16384 | prefill | select | fused | 3.530x | 3.494-3.570x | 3.450-3.593x | 32.804 → 9.309 | 52→16 |
| fp16 | 16384 | prefill | select_read | fused | 1.184x | 1.183-1.184x | 1.177-1.186x | 143.199 → 121.165 | 55→19 |
| fp16 | 32768 | decode_b1 | select | fused | 1.367x | 1.357-1.416x | 1.342-1.443x | 0.514 → 0.369 | 34→16 |
| fp16 | 32768 | decode_b1 | select_read | fused | 1.218x | 1.189-1.277x | 1.188-1.295x | 0.308 → 0.253 | 37→19 |
| fp16 | 32768 | decode_ragged | select | fused | 1.347x | 1.325-1.379x | 1.311-1.453x | 0.216 → 0.157 | 34→16 |
| fp16 | 32768 | decode_ragged | select_read | fused | 1.200x | 1.185-1.210x | 1.105-1.304x | 0.538 → 0.440 | 37→19 |
| fp16 | 32768 | prefill | select | fused | 2.963x | 2.954-2.986x | 2.938-2.987x | 98.058 → 33.098 | 52→16 |
| fp16 | 32768 | prefill | select_read | fused | 1.222x | 1.222-1.227x | 1.200-1.229x | 353.116 → 287.699 | 55→19 |
