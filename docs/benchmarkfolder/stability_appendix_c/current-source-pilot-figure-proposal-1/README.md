# Current-source QCDn3 figure proposal

This is source only. No figure rendering or data discovery is authorized by this
packet. Before execution, the root agent must verify and independently pin the
closed portable data capsule, then supply its direct point-metrics file and hash
to `render(output_directory, *, point_metrics_file,
expected_point_metrics_sha256, plotter_root)`. The data path and hash remain
unassigned until that admission. The output directory must be new.

The wrapper pins plotter-source-6 index
`a0521f17a658e04f8dc320509e733de8fd0e93ac4d02481d908373291ac706ae`
and module
`2fc1d3fb368d623726bb26a55ca6b85e207eb6ebaf07b97da1f1846a6e92395a`.
It preserves all statistical functions, values, provenance, fixed denominators,
strict exceedance comparisons, unavailable bounds, exact breakpoints and Ninja
ratio masking. Only the selected case's display title changes. The full input
dataset SHA remains attached to its statistical JSON.

The existing renderer writes PDF, PNG and JSON. A temporary save hook exports the
same Figure to SVG immediately after its PDF export, using SVG-only fixed hash
salt and timestamp-free metadata. The hook is restored even on failure. This
changes neither the plotting calculations nor the other exports. The filename
`current-source-qcd-n3-raw-dp-stability` identifies this separate source pilot;
earlier figures and canonical files remain untouched.

The retained caption states: current-source QCD pilot; host forced-DP and
card-selected backends; code 110 does not identify the actual reducer or internal
precision; 100 cut-qualified points imply 1% empirical resolution; dotted upper
bounds describe unavailable outcomes rather than measured failure fractions;
incomplete populations and zero Ninja tails mask ratios; the diagnostic is not
independent true error and provides no performance claim. The report should use
`caption.txt` alongside the figure, rather than crowding its existing footer.

Tests use only synthetic data and fake figure objects. They do not import
Matplotlib, render a figure, read a campaign output or execute a native program.
