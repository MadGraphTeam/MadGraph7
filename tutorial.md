## MadGraph 7 tutorial

### Installation

First check out the MadGraph7 repository:
```sh
git clone git@github.com:MadGraphTeam/MadGraph7.git
cd MadGraph7
```

Then install the pre-compiled `madspace` package using
```sh
pip install madspace
```
If the binary distribution does not work for you, you can also build it from source
```sh
pip install scikit_build_core
cd madspace
pip install .
```

For Parton Distribution Functions (PDFs), you need to either:
- install `lhapdf6` through the MadGraph CLI
- install the [python package `lhapdf-management`](https://pypi.org/project/lhapdf-management/) via `pip`:
  ```bash
  pip install lhapdf-management
  ```
  and then set, e.g., `LHAPDF_DATA_PATH` environment variable:
  ```bash
  mkdir -p pdfs
  export LHAPDF_DATA_PATH="$(pwd)/pdfs"
  ```
  finally, install the PDF *NNPDF23_lo_as_0130_qed*
  ```bash
  lhapdf-management install NNPDF23_lo_as_0130_qed
  ```

If you want to try out the `madboard` web interface, you can install it using
```sh
pip install madboard
```

### CPU usage

Open the MadGraph shell with `bin/mg5_aMC`. Then use
```
generate g g > t t~ g
output mg7_simd your_process_name
launch
```
After typing launch, you can choose to edit the `run_card.toml`. Editing parameters using the `set` command is not yet possible.

### CUDA usage

Open the MadGraph shell with `bin/mg5_aMC`. Then use
```
generate g g > t t~ g
output mg7_cuda your_process_name
launch
```
You then have to edit a few entries in the `run_card.toml` by hand:
```toml
# in section [run]
device = "cuda"
thread_pool_size = 1

# in section [generation]
batch_size = 64000
```

### MadBoard

To use MadBoard, go to the directory with you process folders and run
```sh
madboard
```
This will start a server and open MadBoard in your browser.
