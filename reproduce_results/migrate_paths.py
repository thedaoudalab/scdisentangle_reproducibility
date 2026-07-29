import re, glob, nbformat

DATA = re.compile(r"(f?)(['\"])/data/Experiments/Benchmark/(?:SCDISENTANGLE_REPRODUCE|scdisentangle)([^'\"]*)\2")
FIG  = re.compile(r"(f?)(['\"])/data/scDisentangle_paper_figures([^'\"]*)\2")
SETUP = ("import os\n"
         "FIG_ROOT = os.environ.get('SCDIS_FIG', os.path.join(os.environ['SCDIS_ROOT'], 'figures'))")

def repl_data(m):
    q = m.group(2); inner = "'" if q == '"' else '"'
    return f'f{q}{{os.environ[{inner}SCDIS_ROOT{inner}]}}{m.group(3)}{q}'
def repl_fig(m):
    q = m.group(2)
    return f'f{q}{{FIG_ROOT}}{m.group(3)}{q}'

for path in glob.glob('**/*.ipynb', recursive=True):
    if 'ipynb_checkpoints' in path: continue
    nb = nbformat.read(path, as_version=4)
    changed = False
    for c in nb.cells:
        if c.cell_type != 'code': continue
        new = FIG.sub(repl_fig, DATA.sub(repl_data, c.source))
        if new != c.source:
            c.source = new; changed = True
    if changed:
        first = nb.cells[0]
        if not (first.cell_type == 'code' and 'FIG_ROOT = os.environ' in first.source):
            nb.cells.insert(0, nbformat.v4.new_code_cell(SETUP))   # gets a valid id
        nbformat.write(nb, path)
        print('migrated:', path)

