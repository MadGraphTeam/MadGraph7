"""Reproduce private patched source from pinned base. No generated/native work."""
from pathlib import Path
import difflib
import hashlib
import json

HERE=Path(__file__).resolve().parent

def once(s,a,b):
    if s.count(a)!=1:raise ValueError((s.count(a),a[:90]))
    return s.replace(a,b)


def build():
    p='madgraph/iolibs/helas_call_writers.py'
    s=(HERE/'base'/p).read_text()
    start=s.index('    def get_loop_CT_calls(',s.index('class FortranUFOHelasCallWriterOptimized'))
    end=s.index('    def generate_external_wavefunction',start)
    body=s[start:end]
    body=once(body,'res.append(self.get_amplitude_call(lamp_ref))','res.append(invariant_provenance.register(matrix_element, lamp_ref,\n                    self.get_amplitude_call(lamp_ref)))')
    body=once(body,'res.append(self.get_amplitude_call(lamp))','res.append(invariant_provenance.register(matrix_element, lamp,\n                        self.get_amplitude_call(lamp)))')
    body=once(body,'        res = []\n','        from madgraph.loop import invariant_provenance\n        res = []\n')
    (HERE/'src'/p).write_text(s[:start]+body+s[end:])
    p='madgraph/loop/loop_exporters.py';s=(HERE/'base'/p).read_text()
    # Private opt-in: absent option preserves existing source behavior.
    s=once(s,"                'UseDensity': UseDensity}","                'UseDensity': UseDensity,\n                'InvariantProvenance': bool((self.opt.get('invariant_provenance', False) and not self.opt.get('vector_size', 0)))}")
    marker='        self.finalize_dp_terminal_updates()\n'
    s=once(s,marker,marker+'''        if (self.opt.get('invariant_provenance', False) and not self.opt.get('vector_size', 0)):
            from madgraph.loop.invariant_finalize import finalize_pending
            finalize_pending(self)
''')
    marker='''        self.set_group_loops(matrix_element)

        # Initialize a general replacement dictionary'''
    s=once(s,marker,'''        self.set_group_loops(matrix_element)
        if (self.opt.get('invariant_provenance', False) and not self.opt.get('vector_size', 0)):
            from madgraph.loop import invariant_provenance
            invariant_provenance.start(matrix_element)
            self._invariant_pending = getattr(self, '_invariant_pending', {})
            self._invariant_pending[os.path.abspath('.')] = matrix_element

        # Initialize a general replacement dictionary''')
    # When rendering an optimized reduction batch, add only explicit local inputs.
    marker="            new_helascalls_file = new_helascalls_file % helascalls_replace_dict\n"
    s=once(s,marker,'''            if entry_name == 'loop_CT_calls' and context.get('InvariantProvenance'):
                from madgraph.loop.invariant_render import batch_template
                new_helascalls_file = batch_template(new_helascalls_file)
'''+marker)
    marker="        replace_dict[entry_name]='\\n'.join(helascalls_calls)\n"
    s=once(s,marker,'''        if entry_name == 'loop_CT_calls' and context.get('InvariantProvenance'):
            helascalls_calls = [c[:-1]+',GP_ENV,GP_FRESH)' for c in helascalls_calls]
'''+marker)
    # Template rewrites stay after MP call conversion; no accidental MP_ C probe.
    start=s.index('    def write_loopmatrix(',s.index('class LoopProcessOptimizedExporterFortranSA'))
    tail=s[start:]
    reset_marker='        # Set lowercase/uppercase Fortran code\n'
    reset_at=tail.index(reset_marker)
    tail=tail[:reset_at]+'''        from madgraph.loop import invariant_provenance
        if self.opt.get('invariant_provenance', False) and not self.opt.get('vector_size', 0):
            invariant_provenance.start(matrix_element)
        else:
            matrix_element._invariant_provenance = None

'''+tail[reset_at:]
    marker="        # Setup the contextual environment which is used in the splitting\n"
    tail=once(tail,marker,'''        if (self.opt.get('invariant_provenance', False) and not self.opt.get('vector_size', 0)):
            from madgraph.loop.invariant_render import main_template
            file = main_template(file)

'''+marker)
    s=s[:start]+tail
    start=s.index('    def write_mp_compute_loop_coefs(');end=s.index('    def ',start+8)
    part=s[start:end]
    marker="        # Setup the contextual environment which is used in the splitting\n"
    part=once(part,marker,'''        if (self.opt.get('invariant_provenance', False) and not self.opt.get('vector_size', 0)):
            from madgraph.loop.invariant_render import mp_template
            file = mp_template(file)

'''+marker)
    s=s[:start]+part+s[end:]
    # Emit an additional caller, leaving every legacy LOOP body byte-identical.
    marker="            file = misc.apply_template(file, replace_dict)\n            files.append(file)\n"
    s=once(s,marker,'''            if optimized_output and (self.opt.get('invariant_provenance', False) and not self.opt.get('vector_size', 0)):
                from madgraph.loop.invariant_render import wrapper
                private_caller = wrapper(file, replace_dict)
            else:
                private_caller = ''
            file = misc.apply_template(file, replace_dict)
            files.append(file)
            if private_caller:
                files.append(misc.apply_template(private_caller, replace_dict))
''')
    (HERE/'src'/p).write_text(s)
    # Standalone/OLP object inventory and module ordering; no build is performed.
    p='Template/loop_material/StandAlone/SubProcesses/makefile'
    s=(HERE/'base'/p).read_text()
    s=once(s,'\nPROCESS= MadLoopParamReader.o MadLoopCommons.o \\\n','\nPROCESS= MadLoopParamReader.o MadLoopCommons.o \\\n $(patsubst %.f90,%.o,$(wildcard invariant_bounds.f90)) \\\n $(patsubst %.c,%.o,$(wildcard invariant_environment.c)) \\\n')
    s=once(s,'\nOLP_PROCESS= MadLoopParamReader.o MadLoopCommons.o \\\n','\nOLP_PROCESS= MadLoopParamReader.o MadLoopCommons.o \\\n $(patsubst %.f90,%.o,$(wildcard $(LOOP_PREFIX)*/invariant_bounds.f90)) \\\n $(patsubst %.c,%.o,$(wildcard $(LOOP_PREFIX)*/invariant_environment.c)) \\\n')
    s+='''
# Private descriptor plumbing; no new sources means no additional prerequisite.
GP_MODULES := $(patsubst %.f90,%.o,$(wildcard invariant_bounds.f90 $(LOOP_PREFIX)*/invariant_bounds.f90))
GP_INCLUDE := $(addprefix -I,$(sort $(dir $(GP_MODULES))))
FFLAGS += $(GP_INCLUDE)
GP_CONSUMERS := $(filter-out $(GP_MODULES) %invariant_environment.o,$(PROCESS) $(OLP_PROCESS))
$(GP_CONSUMERS): $(GP_MODULES)
%invariant_bounds.o: %invariant_bounds.f90 %invariant_bounds_data.inc
	$(FC) $(filter-out -ffixed-form,$(FFLAGS)) -ffree-form -I$(dir $<) -J$(dir $@) -c $< -o $@
%invariant_environment.o: %invariant_environment.c
	$(CC) $(CFLAGS) -c $< -o $@
'''
    (HERE/'src'/p).write_text(s)
    # Fresh rendered template examples expose transformations without making
    # those templates silently apply to non-opt-in exporters.
    import importlib.util
    spec=importlib.util.spec_from_file_location('render',HERE/'src/madgraph/loop/invariant_render.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    out=HERE/'rendered-templates';out.mkdir(exist_ok=True)
    td='madgraph/iolibs/template_files/loop_optimized/'
    for n,func in [('loop_matrix_standalone.inc',mod.main_template),('mp_compute_loop_coefs.inc',mod.mp_template),('helas_calls_split.inc',mod.batch_template)]:
        (out/n).write_text(func((HERE/'base'/td/n).read_text()))
    patch=[]
    for path in sorted((HERE/'src').rglob('*')):
        if not path.is_file():continue
        name=str(path.relative_to(HERE/'src'));base=HERE/'base'/name
        patch.extend(difflib.unified_diff(base.read_text().splitlines(True) if base.exists() else [],path.read_text().splitlines(True),fromfile='a/'+name if base.exists() else '/dev/null',tofile='b/'+name))
    (HERE/'01-metadata-caller.patch').write_text(''.join(patch))

if __name__=='__main__':build()
