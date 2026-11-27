// SPDX-License-Identifier: 0BSD

//! A native caller exercises the same buffers used by the Python binding.

use rusticol_loop_reduction::runtime::Batch;

#[test]
fn standalone_carrier_buffers_preserve_rows_and_returned_values() {
    for (mode, precision, bits) in [
        ("double", None, 53),
        ("double-double", None, 106),
        ("arbitrary", Some(60), 200),
    ] {
        let batch = Batch::from_decimal(
            ["1", "2", "3", "4"]
                .map(|value| [value.to_owned(), "0".into()])
                .to_vec(),
            2,
            2,
            mode,
            precision,
            Some(vec![91, 4]),
        )
        .unwrap();
        assert_eq!(batch.precision_bits(), bits);
        let original = batch.to_decimal();
        let selected = batch.take_columns(vec![1]).unwrap();
        let joined = Batch::concat_columns(&[&selected, &batch]).unwrap();
        let reversed = joined.take_rows(vec![1, 0], None).unwrap();
        assert_eq!(reversed.row_ids(), [4, 91]);
        let expected = [3, 2, 3, 1, 0, 1].map(|i| original[i].clone());
        assert_eq!(reversed.to_decimal(), expected);
        let regrouped = reversed.reshape_rows(1, 6, vec![12]).unwrap();
        assert_eq!(regrouped.to_decimal(), expected);
        assert!(regrouped.nonfinite_rows().is_empty());
        assert!(batch.take_rows(vec![2], None).is_err());
        assert!(Batch::concat_columns(&[&batch, &reversed]).is_err());
        assert_eq!(batch.to_decimal(), original);
    }
}
