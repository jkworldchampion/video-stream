#!/usr/bin/env python3
import json

clip = json.load(open('output/edge_hf/clip_edge_hf.json'))['summary']
stream = json.load(open('output/edge_hf/stream_edge_hf.json'))['summary']

print('=' * 70)
print('Edge F1 + High-Frequency Analysis: CLIP vs STREAM')
print('=' * 70)
print()
print(f'{"Metric":<25} {"CLIP":>12} {"STREAM":>12} {"Diff (%)":>12}')
print('-' * 70)

metrics = [
    ('Edge F1', 'edge_f1_mean'),
    ('Laplacian Variance', 'lap_var_mean'),
    ('Gradient Mean', 'grad_mean'),
    ('Gradient P95', 'grad_p95'),
    ('FFT HF Ratio', 'fft_hf_ratio_mean')
]

for name, key in metrics:
    c = clip[key]
    s = stream[key]
    diff_pct = ((s - c) / c) * 100
    print(f'{name:<25} {c:>12.6f} {s:>12.6f} {diff_pct:>11.2f}%')

print('=' * 70)
print()
print('해석:')
print('  • Edge F1: 높을수록 경계 보존 능력 우수')
print('  • Laplacian Var: 높을수록 세밀한 디테일 보존')
print('  • Gradient Mean/P95: 높을수록 선명도 유지')
print('  • FFT HF Ratio: 높을수록 고주파 성분(디테일) 보존')
print()
print('결론:')
if stream['edge_f1_mean'] > clip['edge_f1_mean']:
    print(f"  ✓ STREAM이 CLIP보다 Edge F1이 {((stream['edge_f1_mean']-clip['edge_f1_mean'])/clip['edge_f1_mean']*100):.2f}% 높음")
    print('    → STREAM이 경계 보존 능력이 더 우수!')
else:
    print(f"  ✗ CLIP이 STREAM보다 Edge F1이 {((clip['edge_f1_mean']-stream['edge_f1_mean'])/stream['edge_f1_mean']*100):.2f}% 높음")
    print('    → CLIP이 경계 보존 능력이 더 우수')

if stream['lap_var_mean'] > clip['lap_var_mean']:
    print(f"  ✓ STREAM이 CLIP보다 Laplacian Var가 {((stream['lap_var_mean']-clip['lap_var_mean'])/clip['lap_var_mean']*100):.2f}% 높음")
    print('    → STREAM이 세밀한 디테일을 더 잘 보존!')
else:
    print(f"  ✗ CLIP이 STREAM보다 Laplacian Var가 {((clip['lap_var_mean']-stream['lap_var_mean'])/stream['lap_var_mean']*100):.2f}% 높음")
    print('    → CLIP이 세밀한 디테일을 더 잘 보존')

print()
