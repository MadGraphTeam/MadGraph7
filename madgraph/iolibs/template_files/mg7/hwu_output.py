"""The event-sample histograms in the HwU format aMC@NLO writes.

``EventHistograms::to_json()`` (stored in ``Events/<run>/info.json`` under
"event_histograms") already holds everything a plot needs: the nominal
distribution, one distribution per systematics variation, and the scale/PDF
bands built from them. HwU is the format the rest of MadGraph reads -- an
aMC@NLO run writes ``MADatNLO.HwU`` and ``madgraph/various/histograms.py``
plots and compares those -- so writing the same thing here is what lets an
mg7 run be overlaid on an NLO one without a conversion step in between.

Two conventions of the format are worth stating, because they are where a
converter goes wrong:

* HwU bins hold a **differential** cross section, dsigma/dx: the value written
  is the cross section in the bin divided by the bin width, so that
  ``sum(central * width)`` is the cross section. The madspace histograms hold
  the cross section *in* the bin, so every column is divided by the width here.
* HwU has **no under/overflow bins**. The madspace arrays carry them (first and
  last entry), and they are dropped, exactly as an aMC@NLO analysis does. The
  integral of an HwU plot is therefore the cross section inside the range, not
  the total one.

The weight columns are labelled the way ``HwU.weight_label_scale`` and
``HwU.weight_label_PDF`` expect ("muR=1.00 muF=2.00", "PDF= 331901"), since
that is what makes histograms.py recognise them as a scale envelope and a PDF
band rather than as anonymous extra curves.

This module imports nothing: it is a pure transformation of the two dicts
madspace writes, so anything holding them can produce an HwU file.
"""


def _float(value):
    """One HwU number: signed scientific notation, as aMC@NLO writes it."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 0.
    if value != value:  # NaN, which no HwU reader expects in a bin
        value = 0.
    return '%+.7e' % value


def weight_labels(summary):
    """{weight id: HwU column label} from madspace's systematics summary.

    The group an id belongs to has the last word, not the variation's own
    fields: the central member of a PDF set carries mur = muf = 1 and no
    pdf_index, and labelling it as a scale variation would both hide it from
    the PDF band and duplicate the nominal column.
    """

    labels = {}
    if not summary:
        return labels
    variations = {}
    for entry in summary.get('variations', []):
        if 'id' in entry:
            variations[entry['id']] = entry

    for wgt_id in (summary.get('scale') or {}).get('weight_ids', []):
        entry = variations.get(wgt_id, {})
        mur = entry.get('mur', 1.)
        muf = entry.get('muf', 1.)
        dyn = entry.get('dyn', -1)
        if dyn is not None and dyn != -1:
            labels[wgt_id] = 'dyn=%d muR=%.2f muF=%.2f' % (dyn, mur, muf)
        else:
            labels[wgt_id] = 'muR=%.2f muF=%.2f' % (mur, muf)

    for group in summary.get('pdf') or []:
        for wgt_id in group.get('weight_ids', []):
            entry = variations.get(wgt_id, {})
            lhaid = entry.get('pdf_lhaid', group.get('pdf_lhaid'))
            member = entry.get('pdf_member', 0)
            if lhaid is None:
                continue
            # the LHAPDF id of the member itself, which is what an HwU
            # "PDF= <id>" column names
            labels[wgt_id] = 'PDF= %d' % (int(lhaid) + int(member))

    return labels


def _columns(histogram, labels):
    """(column label, values) for one histogram, under/overflow still in."""

    columns = [('central value', histogram.get('bin_values') or []),
               ('dy', histogram.get('bin_errors') or [])]

    envelope = histogram.get('scale_envelope')
    if envelope:
        columns.append(('delta_mu_min @aux', envelope.get('low') or []))
        columns.append(('delta_mu_max @aux', envelope.get('high') or []))

    for group in histogram.get('pdf_uncertainty') or []:
        central = group.get('central') or []
        down = group.get('uncertainty_down') or []
        up = group.get('uncertainty_up') or []
        if not central:
            continue
        columns.append(('delta_pdf_min @aux',
                        [c - d for c, d in zip(central, down)]))
        columns.append(('delta_pdf_max @aux',
                        [c + u for c, u in zip(central, up)]))
        break  # the format has one PDF band; further sets stay as raw columns

    for index, entry in enumerate(histogram.get('weights') or []):
        label = labels.get(entry.get('id'), 'wgt_%d' % (index + 1))
        columns.append((label, entry.get('bin_values') or []))

    return columns


def to_hwu(event_histograms, summary=None, type_label='LO'):
    """Render madspace's event-sample histograms as HwU text.

    `event_histograms` is the "event_histograms" list of info.json and
    `summary` its "systematics" dict (None when the run had none).
    `type_label` is the HwU TYPE tag, which is how histograms.py tells the
    curves of two runs apart when it draws them together.
    """

    histograms = [h for h in event_histograms or [] if h.get('bin_count')]
    if not histograms:
        return ''
    labels = weight_labels(summary)

    header = ['xmin', 'xmax']
    header += [label for label, _ in _columns(histograms[0], labels)]
    lines = ['##& ' + ' & '.join(header), '']

    for histogram in histograms:
        bin_count = int(histogram['bin_count'])
        low = float(histogram['min'])
        width = (float(histogram['max']) - low) / bin_count
        columns = _columns(histogram, labels)
        lines.append('<histogram> %d "%s |X_AXIS@LIN |Y_AXIS@LOG |TYPE@%s"'
                     % (bin_count, histogram.get('name', ''), type_label))
        for index in range(bin_count):
            row = [_float(low + index * width), _float(low + (index + 1) * width)]
            for _label, values in columns:
                # index + 1: the madspace arrays start with the underflow bin
                value = values[index + 1] if index + 1 < len(values) else 0.
                row.append(_float(value / width if width else 0.))
            lines.append('  ' + '   '.join(row))
        lines.append('<\\histogram>')
        lines.append('')

    return '\n'.join(lines)
