// SPDX-License-Identifier: 0BSD

//! pyAmpliCol's Symbolica library key in this crate: the thread-bound guards
//! of the `pyamplicol` crate, which registers the key (see there).

pub(crate) use pyamplicol::{unlock, unlock_thread};
