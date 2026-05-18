import os
import re
import glob

py_scripts = glob.glob("/home/work/mlp/hslim/LASEF2/scripts/src/eval_*.py")
sh_scripts = glob.glob("/home/work/mlp/hslim/LASEF2/scripts/eval/eval_*.sh")

print(f"Found {len(py_scripts)} python scripts")
print(f"Found {len(sh_scripts)} shell scripts")

for py_path in py_scripts:
    with open(py_path, 'r') as f:
        content = f.read()

    # 1. Add args.rollout
    if 'parser.add_argument("--rollout"' not in content:
        content = content.replace('parser.add_argument("--dtype", type=str, default="bfloat16")',
                                  'parser.add_argument("--dtype", type=str, default="bfloat16")\n    parser.add_argument("--rollout", type=int, default=1)')

    # 2. Add eff variables
    if 'eff_temp = 0.7' not in content:
        content = content.replace('tok = llm.get_tokenizer()',
                                  'tok = llm.get_tokenizer()\n    \n    eff_temp = 0.7 if args.rollout > 1 else args.temp\n    eff_top_p = 0.95 if args.rollout > 1 else args.top_p')

    # 3. Replace kwargs in SamplingParams
    content = content.replace('temperature=args.temp', 'temperature=eff_temp')
    content = content.replace('top_p=args.top_p', 'top_p=eff_top_p')
    content = content.replace('temperature=0.6', 'temperature=eff_temp')
    content = content.replace('top_p=0.95', 'top_p=eff_top_p')

    # 4. Add n=args.rollout to gen_sp (avoiding trans_sp and sk_sp)
    def replace_gen_sp(match):
        inner = match.group(1)
        if re.search(r'\bn=\d+', inner):
            inner = re.sub(r'\bn=\d+', 'n=args.rollout', inner)
        else:
            if inner.rstrip().endswith(','):
                inner = inner.rstrip() + ' n=args.rollout'
            else:
                inner = inner + ', n=args.rollout'
        return f"gen_sp = SamplingParams({inner})"

    content = re.sub(r'gen_sp = SamplingParams\((.*?)\)', replace_gen_sp, content, flags=re.DOTALL)

    with open(py_path, 'w') as f:
        f.write(content)
    print(f"Patched {py_path}")

for sh_path in sh_scripts:
    with open(sh_path, 'r') as f:
        content = f.read()
    
    if 'local ROLLOUT_FLAG=""' not in content:
        inject_code = """
    local ROLLOUT_FLAG=""
    local ROLLOUT_SUFFIX=""
    if [[ -n "$ROLLOUT" ]] && [[ "$ROLLOUT" -gt 1 ]]; then
        ROLLOUT_FLAG="--rollout $ROLLOUT"
        ROLLOUT_SUFFIX="_rollout${ROLLOUT}"
    fi
"""
        content = re.sub(r'(\s*local OUTPUT=)', r'\n' + inject_code + r'\1', content)
        content = re.sub(r'(local OUTPUT=.*)\.jsonl', r'\1${ROLLOUT_SUFFIX}.jsonl', content)

    if '$ROLLOUT_FLAG \\' not in content:
        content = content.replace('--sample_ratio $SAMPLING \\', '--sample_ratio $SAMPLING \\\n        $ROLLOUT_FLAG \\')
        
    with open(sh_path, 'w') as f:
        f.write(content)
    print(f"Patched {sh_path}")
