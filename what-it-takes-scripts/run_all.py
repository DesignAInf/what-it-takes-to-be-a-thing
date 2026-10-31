"""Run every verification script of 'What It Takes to Be a Thing' and compare with the reference output.
Usage: python run_all.py            (runs all scripts, prints a summary)
       python run_all.py --write    (regenerates expected_output.txt)
Floating-point digits may differ in the last places across library versions; the comparison ignores numbers
smaller than 1e-12 in absolute value and compares the rest to 6 significant digits."""
import subprocess, sys, re, pathlib
here=pathlib.Path(__file__).parent
scripts=sorted(p.name for p in here.glob("s[0-9][0-9]_*.py"))
num=re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
def norm(t):
    def r(m):
        v=float(m.group(0))
        return "0" if abs(v)<1e-12 else "%.6g"%v
    return num.sub(r,t)
out={}
for s in scripts:
    r=subprocess.run([sys.executable,s],cwd=here,capture_output=True,text=True)
    out[s]=r.stdout.strip()
    print("%-36s %s"%(s,"ok" if r.returncode==0 else "ERROR"))
text="".join("### %s\n%s\n"%(s,out[s]) for s in scripts)
ref=here/"expected_output.txt"
if "--write" in sys.argv or not ref.exists():
    ref.write_text(text); print("reference output written to",ref.name)
else:
    blocks=dict(re.findall(r"### (\S+)\n(.*?)(?=\n### |\Z)",ref.read_text(),flags=re.S))
    bad=[s for s in scripts if norm(blocks.get(s,"").strip())!=norm(out[s])]
    print("all outputs match the reference" if not bad else "outputs differ from the reference: "+", ".join(bad)); sys.exit(1 if bad else 0)
