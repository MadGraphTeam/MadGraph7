"""Persist open incoming/outgoing spin tensors and contract prepared inputs.

The combined matrix index is a + incoming_dim*h. These unnormalised tensors
describe scattering; their incoming labels are not extra final-state spins.
Only the incoming spin average is removed. Native colour averages, final-state
symmetry divisors and sums over other spectator spins remain in the arrays.
The Goldstone-relative comparison array 'relative' is optional. The arrays
are Hermitian in their two matrix indices; saved tensors keep only the upper
triangles, which is lossless up to rounding.
"""

import base64
import io
import json
import re
from xml.etree import ElementTree

try:
    import numpy as np
except ImportError:  # only the tensor functions need numpy
    np = None


_ARRAYS = ('born', 'correction', 'relative')
_OPTIONAL = ('relative',)
# Variant order of a tensor without a 'variants' entry
_VARIANTS = ('qed', 'weak', 'sr_off', 'sr_and_floor_off', 'floor_off')
_RECORD = re.compile(r'<sudakov_process_tensor\b[^>]*>.*?</sudakov_process_tensor>', re.S)


def _arrays(tensor):
    """Names of the arrays the tensor holds."""
    return [name for name in _ARRAYS if name not in _OPTIONAL or name in tensor]


def _validate(tensor):
    if tensor.get('version') != 1:
        raise ValueError('Unsupported process tensor version')
    ni, no = int(tensor['incoming_dim']), int(tensor['outgoing_dim'])
    if ni < 1 or no < 1:
        raise ValueError('Invalid process tensor dimensions')
    dimension = ni * no
    nvariants = len(tensor.get('variants') or _VARIANTS)
    for name in _arrays(tensor):
        array = np.asarray(tensor[name])
        shape = (dimension, dimension) + (() if name == 'born' else (nvariants,))
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError('Invalid process tensor array: ' + name)
    return ni, no


def contract_process_tensor(tensor, incoming_density=None):
    """Return unnormalised outgoing Born and linear-correction matrices.

    rho_in[a,b] multiplies R[h,a;k,b], with no conjugation or transposition
    of rho_in. Its default is the unpolarised I/n. An explicitly prepared
    input must be Hermitian, positive semidefinite, and have unit trace.
    The returned EW expansion is not projected onto positive matrices.
    density_relative is returned only for a tensor with a relative array.
    """
    ni, no = _validate(tensor)
    rho = np.eye(ni, dtype=complex) / ni if incoming_density is None else np.asarray(incoming_density, complex)
    if (rho.shape != (ni, ni) or not np.isfinite(rho).all() or
            not np.allclose(rho, rho.conj().T, rtol=0, atol=1e-12) or
            not np.isclose(np.trace(rho), 1, rtol=0, atol=1e-12) or
            np.linalg.eigvalsh(rho).min() < -1e-12):
        raise ValueError('Incoming spin density must be Hermitian, positive and unit trace')
    result = {}
    for name in _arrays(tensor):
        array = np.asarray(tensor[name])
        blocks = array.reshape((no, ni, no, ni) + array.shape[2:])
        result['density_' + name] = np.einsum('iajb...,ab->ij...', blocks, rho)
    return result


def save_process_tensor(destination, tensor):
    """Save to a path or binary stream as compressed NPZ, without pickles.

    Only the upper triangle of each Hermitian array is stored.
    """
    ni, no = _validate(tensor)
    upper = np.triu_indices(ni * no)
    metadata = {key: value for key, value in tensor.items() if key not in _ARRAYS}
    metadata['packing'] = 'upper_triangle'
    np.savez_compressed(destination, metadata_json=json.dumps(metadata, sort_keys=True),
                        **{name: np.asarray(tensor[name])[upper] for name in _arrays(tensor)})


def load_process_tensor(source):
    """Read a path or binary stream saved by save_process_tensor."""
    with np.load(source, allow_pickle=False) as data:
        tensor = json.loads(str(data['metadata_json']))
        arrays = {name: np.array(data[name], copy=True)
                  for name in _ARRAYS if name in data.files}
    if tensor.pop('packing', None) == 'upper_triangle':
        dimension = int(tensor['incoming_dim']) * int(tensor['outgoing_dim'])
        upper = np.triu_indices(dimension)
        lower = np.tril_indices(dimension, -1)
        for name, packed in arrays.items():
            array = np.zeros((dimension, dimension) + packed.shape[1:], complex)
            array[upper] = packed
            array[lower] = np.conj(array[lower[1], lower[0]])
            arrays[name] = array
    tensor.update(arrays)
    _validate(tensor)
    return tensor


def clear_event_process_tensors(event):
    """Remove a previous scan's tensors before starting a new xi scan."""
    if '<sudakov_process_tensor' in event.tag:
        event.tag = _RECORD.sub('', event.tag)


def set_event_process_tensor(event, tensor, xi_index, xi, **context):
    """Keep one lossless record per xi inside the LHE event's extension tags.

    A None tensor records that this event/xi used an existing scalar or unit
    path. The explicit status prevents missing tensors from looking like zero
    matrices. Re-evaluating the same xi replaces its record.
    """
    xi_index = int(xi_index)
    def retain(match):
        key = re.search(r'\bxi_index="(\d+)"', match.group())
        return '' if key and int(key.group(1)) == xi_index else match.group()
    tag = event.tag
    if 'xi_index="%d"' % xi_index in tag:
        tag = _RECORD.sub(retain, tag)
    if tensor is None:
        attrs = 'status="not_evaluated"'
        payload = ''
    else:
        tensor = dict(tensor, xi_index=xi_index, xi=float(xi), **context)
        stream = io.BytesIO()
        save_process_tensor(stream, tensor)
        encoded = base64.b64encode(stream.getvalue()).decode('ascii')
        payload = '\n'.join(encoded[i:i + 120] for i in range(0, len(encoded), 120))
        attrs = 'status="available" encoding="base64-npz"'
    event.tag = tag.rstrip() + ('\n<sudakov_process_tensor version="1" xi_index="%d" '
                               'xi="%s" %s>\n%s\n</sudakov_process_tensor>\n' %
                               (xi_index, repr(float(xi)), attrs, payload))


def read_event_process_tensors(event):
    """Return records in xi order; unavailable records contain status metadata."""
    text = event.tag if hasattr(event, 'tag') else str(event)
    records = []
    for match in _RECORD.finditer(text):
        element = ElementTree.fromstring(match.group())
        if element.attrib.get('version') != '1':
            raise ValueError('Unsupported LHE process tensor version')
        xi_index = int(element.attrib['xi_index'])
        xi = float(element.attrib['xi'])
        status = element.attrib['status']
        if status == 'not_evaluated':
            tensor = dict(version=1, xi_index=xi_index, xi=xi)
        elif status == 'available' and element.attrib.get('encoding') == 'base64-npz':
            payload = ''.join((element.text or '').split())
            tensor = load_process_tensor(io.BytesIO(base64.b64decode(payload, validate=True)))
            if tensor.get('xi_index') != xi_index or tensor.get('xi') != xi:
                raise ValueError('LHE process tensor metadata does not match its tag')
        else:
            raise ValueError('Invalid LHE process tensor status or encoding')
        tensor['status'] = status
        records.append(tensor)
    return sorted(records, key=lambda tensor: tensor['xi_index'])
