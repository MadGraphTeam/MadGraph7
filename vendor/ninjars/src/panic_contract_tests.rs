//! Boundary recovery without creating a reducer, context, or Symbolica worker.
use super::*;

#[test]
fn panic_is_contained_and_success_clears_calling_thread_error() {
    assert_eq!(guard(|| Err("earlier rejected request".into())), ERROR);
    assert_eq!(guard(|| panic!("deliberate boundary regression panic")), ERROR);

    let required = unsafe { ninjars_last_error(std::ptr::null_mut(), 0) };
    assert!(required > 1);
    let mut message = vec![b'!'; required + 3];
    assert_eq!(
        unsafe { ninjars_last_error(message.as_mut_ptr(), required) },
        required
    );
    assert_eq!(message[required - 1], 0);
    assert_eq!(&message[required..], &[b'!'; 3]);
    let text = std::str::from_utf8(&message[..required - 1]).unwrap();
    assert!(text.contains("panic contained at the C boundary"));
    assert!(!text.contains("earlier rejected request"));

    let mut truncated = [b'!'; 8];
    assert_eq!(
        unsafe { ninjars_last_error(truncated.as_mut_ptr(), truncated.len()) },
        required
    );
    assert_eq!(truncated[7], 0);
    assert_eq!(&truncated[..7], &message[..7]);

    assert_eq!(guard(|| Ok(SUCCESS)), SUCCESS);
    assert_eq!(unsafe { ninjars_last_error(std::ptr::null_mut(), 0) }, 1);
    let mut cleared = [b'!'; 4];
    assert_eq!(
        unsafe { ninjars_last_error(cleared.as_mut_ptr(), cleared.len()) },
        1
    );
    assert_eq!(cleared, [0, b'!', b'!', b'!']);
}
