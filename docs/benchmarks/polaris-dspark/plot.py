"""Render the committed comparison data: python plot.py --output /tmp/comparison.png"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent
args = argparse.ArgumentParser()
args.add_argument('--output', type=Path, default=ROOT/'comparison.png')
output = args.parse_args().output
output.parent.mkdir(parents=True, exist_ok=True)
rows = json.loads((ROOT/'results.json').read_text())
assert all(x['success'] == 300 and x['failures'] == 0 for x in rows)
FRAMEWORKS = {'sfllm': ('SFLLM', '#1676bd', 'o'),
              'sglang': ('SGLang (unmodified upstream)', '#e06c26', 's')}
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'svg.fonttype': 'none'})
complete = len(rows) == 18
suffix = '' if complete else ' (in progress)'
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
fig.subplots_adjust(left=0.075, right=0.98, bottom=0.20, top=0.78, wspace=0.29)
for framework, (label, color, marker) in FRAMEWORKS.items():
    series = [x for x in rows if x['framework'] == framework]
    if not series:
        continue
    style = '-' if len(series) == 9 else 'None'
    axes[0].plot([x['concurrency'] for x in series], [x['output_tok_s'] for x in series],
                 label=label, color=color, marker=marker, linestyle=style, linewidth=2, markersize=6)
    # Connect operating points in concurrency order; annotate actual concurrency.
    axes[1].plot([x['output_tok_s'] for x in series], [x['tpot_p95_ms'] for x in series],
                 label=label, color=color, marker=marker, linestyle=style, linewidth=2, markersize=6)
    for x in series:
        axes[1].annotate(str(x['concurrency']), (x['output_tok_s'], x['tpot_p95_ms']),
                         xytext=(4, 5), textcoords='offset points', fontsize=8, color=color)
    peak = max(series, key=lambda x: x['output_tok_s'])
    axes[0].annotate(f"{peak['output_tok_s']:,.0f} tok/s @ C={peak['concurrency']}",
                     (peak['concurrency'], peak['output_tok_s']),
                     xytext=(-3 if peak['concurrency'] >= 48 else 0, 12),
                     textcoords='offset points', ha='right' if peak['concurrency'] >= 48 else 'center',
                     color=color, fontsize=10)
axes[0].set(xlabel='Client concurrency', ylabel='Output throughput (tok/s)',
            title='Concurrency vs. throughput' + suffix, ylim=(0, None), xlim=(0, 68))
axes[0].set_xticks([1, 4, 8, 12, 16, 24, 32, 48, 64])
axes[0].tick_params(axis='x', labelsize=9)
axes[1].set(xlabel='Output throughput (tok/s)', ylabel='P95 TPOT (ms/token)',
            title='Throughput vs. generation latency', ylim=(0, None), xlim=(0, None))
if rows:
    axes[0].set_ylim(0, max(x['output_tok_s'] for x in rows) * 1.17)
    axes[1].set_ylim(0, max(x['tpot_p95_ms'] for x in rows) * 1.14)
for ax in axes:
    ax.grid(alpha=0.2)
axes[0].legend(loc='lower right', fontsize=9)
fig.suptitle('DSpark | Polaris random 300 | 1 x H100 NVL', fontsize=15, y=0.97)
fig.text(0.5, 0.885, 'FP8 target / BF16 activations / FP32 SSM state | identical workload and generation limits',
         ha='center', fontsize=10, color='#555555')
fig.text(0.5, 0.835, 'SFLLM 1755593 | SGLang official main e262b44eef (no local source changes)', ha='center', fontsize=9, color='#555555')
footer = ('Best among recorded candidates at each concurrency; one 300-request formal run per point.\n'
          'Corrected SGLang: native FlashInfer draft attention. TPOT labels = concurrency.')
if not complete:
    measured = ', '.join(str(x['concurrency']) for x in rows if x['framework']=='sglang')
    footer = f'Corrected SGLang measured at C={measured}; other points pending, no interpolation.\nNative FlashInfer draft attention; each shown point = 300 requests.'
fig.text(0.5, 0.045, footer, ha='center', fontsize=9, color='#555555', linespacing=1.5)
for ext in ['png', 'svg', 'pdf']:
    fig.savefig(output.with_suffix('.' + ext), dpi=180)
svg = output.with_suffix('.svg')
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
plt.close(fig)
