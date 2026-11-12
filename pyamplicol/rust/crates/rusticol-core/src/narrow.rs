// SPDX-License-Identifier: 0BSD

//! Checked narrowing of counts, ids and offsets to their compact storage
//! width (`u32` ids keep tables small where memory matters).
//!
//! A value beyond the width is refused with a [`NarrowError`] naming the
//! quantity, the value and the width; it never wraps (release builds have no
//! overflow checks, so an `as` cast would wrap silently).

use crate::{RusticolError, RusticolErrorKind};
use std::fmt;

/// A count, id or offset beyond its storage width.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NarrowError {
    quantity: &'static str,
    value: u128,
    width: &'static str,
}

impl NarrowError {
    pub fn new(quantity: &'static str, value: u128, width: &'static str) -> Self {
        Self {
            quantity,
            value,
            width,
        }
    }

    /// The quantity that did not fit.
    pub fn quantity(&self) -> &'static str {
        self.quantity
    }
}

impl fmt::Display for NarrowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} exceeds the {} range of its storage",
            self.quantity, self.value, self.width
        )
    }
}

impl std::error::Error for NarrowError {}

impl NarrowError {
    /// The error as a [`RusticolError`] (no `From` impl: a second
    /// conversion into `RusticolError` would make `?` ambiguous wherever
    /// the error type is inferred).
    pub fn into_error(self) -> RusticolError {
        RusticolError::with_kind(
            RusticolErrorKind::UnsupportedRuntimeCapability,
            self.to_string(),
        )
    }
}

/// Unsigned integer widths a value narrows to.
pub trait Width: Sized + Copy {
    const NAME: &'static str;
    fn from_u128(value: u128) -> Option<Self>;
}

macro_rules! width {
    ($($t:ty),*) => {$(
        impl Width for $t {
            const NAME: &'static str = stringify!($t);
            fn from_u128(value: u128) -> Option<Self> {
                <$t>::try_from(value).ok()
            }
        }
    )*};
}
width!(u8, u16, u32, u64, usize);

/// `value` as a `W`, or the error naming `quantity`.
#[inline]
pub fn narrow<W: Width>(value: usize, quantity: &'static str) -> Result<W, NarrowError> {
    narrow_u128(value as u128, quantity)
}

/// [`narrow`] of a `u64`.
#[inline]
pub fn narrow_u64<W: Width>(value: u64, quantity: &'static str) -> Result<W, NarrowError> {
    narrow_u128(u128::from(value), quantity)
}

#[inline]
fn narrow_u128<W: Width>(value: u128, quantity: &'static str) -> Result<W, NarrowError> {
    W::from_u128(value).ok_or(NarrowError::new(quantity, value, W::NAME))
}

/// [`narrow`] with the error as a [`RusticolError`].
#[inline]
pub fn checked<W: Width>(value: usize, quantity: &'static str) -> Result<W, RusticolError> {
    narrow(value, quantity).map_err(NarrowError::into_error)
}

/// [`narrow`] with the error as its message.
#[inline]
pub fn checked_str<W: Width>(value: usize, quantity: &'static str) -> Result<W, String> {
    narrow(value, quantity).map_err(|error| error.to_string())
}

/// Whether the stored id `id` names position `index` (an index beyond the
/// id's width never matches; no `as` cast wraps it onto a small id).
#[inline]
pub fn is_index<W: Width + PartialEq>(id: W, index: usize) -> bool {
    W::from_u128(index as u128) == Some(id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_index_past_the_width_names_no_id() {
        assert!(is_index(7_u32, 7));
        assert!(!is_index(7_u32, 8));
        // A wrapping cast would equate these.
        assert!(!is_index(0_u32, 1_usize << 32));
        assert!(!is_index(1_u16, 65_537));
    }

    #[test]
    fn values_past_the_width_are_refused_with_the_quantity() {
        assert_eq!(narrow::<u32>(7, "row").unwrap(), 7);
        assert_eq!(narrow::<u32>(u32::MAX as usize, "row").unwrap(), u32::MAX);
        let error = narrow::<u32>(u32::MAX as usize + 1, "dag current id").unwrap_err();
        assert_eq!(
            error.to_string(),
            "dag current id 4294967296 exceeds the u32 range of its storage"
        );
        assert_eq!(error.quantity(), "dag current id");
        assert!(narrow::<u16>(65_536, "component count").is_err());
        assert!(narrow::<u8>(256, "leg label").is_err());
        assert_eq!(narrow_u64::<u32>(1 << 20, "x").unwrap(), 1 << 20);
        let error = checked::<u32>(usize::MAX, "offset").unwrap_err();
        assert_eq!(
            error.kind(),
            RusticolErrorKind::UnsupportedRuntimeCapability
        );
        assert!(error.message().contains("offset"));
    }
}
