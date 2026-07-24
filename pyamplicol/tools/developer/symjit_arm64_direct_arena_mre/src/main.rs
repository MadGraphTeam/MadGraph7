use std::ptr;

use symjit::{
    Config, DirectApplication, DirectApplicationMetadata, DirectDestinationOperation,
    DirectInputBinding, DirectPlane, DirectScalar, DIRECT_STATUS_OK,
};

// Portable O2 source application from the smallest prepared kernel that
// reliably leaves x0 nonzero in the ARM64 scalar generator.
const SOURCE_APPLICATION_HEX: &str = "\
e9080d419587564003000000000000005d4d8efbae44b2c2702e42c7773c2c806d9220000000000000\
000000000000000c00000000000000040000000000000000000000000000000000000000000000bc\
27e6ab251ef2120000000000000000220100000000000003000000000000000000000000000000c83b\
7f669ea0e63fc83b7f669ea0e6bf000000000000000005000000000000009e0cb023639b6a87a0000\
0000000000005400100054101028e40404103418106824041400440c2900005420100054301018e424\
243034381048242434205440100054001028e4044400344810a2040444042034281008240424003418\
1080542010103c1c2900020424142c10341810220404142400440410005420100054001018e4042408\
04243448240404282404140034281080541010203c1c2900020414241c1034281002040424140044041\
0203000000000000000000000000000000c83b7f669ea0e63fc83b7f669ea0e6bf000000000000000\
0";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source = decode_hex(SOURCE_APPLICATION_HEX)?;
    let metadata = DirectApplicationMetadata::new(
        DirectDestinationOperation::Add,
        vec![],
        (0..12).map(DirectInputBinding::Plane).collect(),
        16,
        2,
        vec![12, 13, 14, 15],
    )?;
    let callable = DirectApplication::from_source_storage(
        &mut source.as_slice(),
        &Config::default(),
        metadata,
    )?
    .seal()?
    .into_callable();

    let mut storage = [[0.0_f64; 1]; 16];
    for (index, plane) in storage.iter_mut().enumerate() {
        plane[0] = (index + 1) as f64 / 10.0;
    }
    let mut descriptors = storage
        .iter_mut()
        .map(|values| unsafe { DirectPlane::from_raw_parts(values.as_mut_ptr(), values.len()) })
        .collect::<Vec<_>>();
    descriptors.extend_from_slice(&[
        descriptors[12],
        descriptors[13],
        descriptors[14],
        descriptors[15],
    ]);

    let factor_re = 1.0;
    let factor_im = 0.0;
    let scalars = [
        unsafe { DirectScalar::from_raw(ptr::from_ref(&factor_re)) },
        unsafe { DirectScalar::from_raw(ptr::from_ref(&factor_im)) },
    ];
    let status = unsafe { callable.handle().invoke(&descriptors, &scalars, 0, 1) };

    println!("target_arch={}", std::env::consts::ARCH);
    println!("status={status}");
    println!(
        "destination=[{:.17e}, {:.17e}, {:.17e}, {:.17e}]",
        storage[12][0], storage[13][0], storage[14][0], storage[15][0]
    );

    if status != DIRECT_STATUS_OK {
        return Err(format!("Direct-Arena scalar call returned status {status}").into());
    }
    Ok(())
}

fn decode_hex(value: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    if value.len() % 2 != 0 {
        return Err("source application hex has odd length".into());
    }
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|digits| {
            let text = std::str::from_utf8(digits)?;
            Ok(u8::from_str_radix(text, 16)?)
        })
        .collect()
}
