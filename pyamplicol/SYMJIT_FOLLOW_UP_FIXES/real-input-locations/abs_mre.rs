//! Included as a test submodule of SymJIT's complexify.rs.
use super::*;
use super::super::mir::{Instruction, UniOp};

#[test]
fn real_abs_uses_the_scalar_register_coordinates() {
    let mut config = Config::default();
    config.set_complex(true);
    let mut lowering = Complexifier::new(&HashSet::new(), config);
    // A real constant avoids any reliance on real-input markers.
    lowering.load_const(Reg::Gen(2), 0);
    lowering.abs(Reg::Gen(3), Reg::Gen(2));
    assert!(matches!(lowering.mir.code.iter().nth(1), Some(Instruction::Uni {
        op: UniOp::Abs,
        dst: Reg::Gen(10), // re(Gen(3)) = Gen(4 + 2*3)
        s1: Reg::Gen(8),  // re(Gen(2)) = Gen(4 + 2*2)
    })));
}
