import copy
from fractions import Fraction as F
import unittest
import audit as a


def c(x,y=0): return F(x),F(y)
P=[[F(1),F(0),F(0),F(0)],[F(0),F(1),F(0),F(0)],[F(0),F(0),F(1),F(0)]]
BOX={i:[F(x) for x in row] for i,row in enumerate(((2,1,3,-1),(3,1,3,-1),(2,2,3,-1),(2,1,4,-1)),1)}


class PureTests(unittest.TestCase):
    def test_nonzero_origin_and_odd_sign(self):
        ell=[F(2),F(3),F(5),F(7)];origin=[F(11),F(13),F(17),F(19)]
        m1=[c(-x) for x in ell];m2=[[c(x*y) for y in ell] for x in ell]
        j,k=a.untranslate(c(1),m1,m2,origin)
        self.assertEqual(j,[c(x+y) for x,y in zip(ell,origin)])
        self.assertEqual(k,[[c((ell[mu]+origin[mu])*(ell[nu]+origin[nu])) for nu in range(4)] for mu in range(4)])

    def test_every_pinch_permutation_reversal_and_translation(self):
        coeff=[c(3,1),c(5,-2)];scalar=c(7,-1);translation=[F(4),F(-3),F(2),F(1)]
        for omitted,(_,perm) in a.PINCHES.items():
            for eta in (1,-1):
                qt={i:[eta*(BOX[parent][mu]-translation[mu]) for mu in range(4)] for i,parent in enumerate(perm,1)}
                p=[[qt[j][mu]-qt[1][mu] for mu in range(4)] for j in (2,3)]
                jm=[a.total(a.scale(p[t][mu],coeff[t]) for t in range(2)) for mu in range(4)]
                moment=[a.sub(a.scale(qt[1][mu],scalar),jm[mu]) for mu in range(4)]
                recovered,residual,_=a.recover_vector_coefficients(moment,qt[1],scalar,p)
                self.assertEqual(recovered,coeff);self.assertEqual(residual,[a.ZERO]*4)
                gathered=a.regather(recovered,scalar,BOX,perm)
                physical=[a.add(a.scale(eta*-1,moment[mu]),a.scale(BOX[1][mu]-translation[mu],scalar)) for mu in range(4)]
                self.assertEqual(gathered,physical)
                routing=a.affine_routing(BOX,qt,perm)
                self.assertEqual(routing['eta'],eta);self.assertTrue(routing['exact_affine_routing'])

    def test_affine_defect_is_not_exact_routing(self):
        perm=(1,2,3);qt={i:BOX[i][:] for i in perm};qt[2][3]+=F(1,10**40)
        self.assertFalse(a.affine_routing(BOX,qt,perm)['exact_affine_routing'])

    def test_unused_coordinate_and_decomposition_residuals_retained(self):
        moment=[c(-2),c(-3),c(0),c(1,1)]
        _,residual,_=a.recover_vector_coefficients(moment,[F(0)]*4,a.ZERO,P[:2])
        self.assertNotEqual(residual[3],a.ZERO)
        k=[[a.ZERO for _ in range(4)] for _ in range(4)];k[0][3]=k[3][0]=c(1)
        _,_,residual,_=a.decompose(k,P)
        self.assertTrue(any(z!=a.ZERO for z in residual))

    def test_metric_piece_not_transformed_by_supplied_gram(self):
        eta=c(3,2);k=[[a.scale(a.METRIC[i] if i==j else 0,eta) for j in range(4)] for i in range(4)]
        metric,b,residual,antisym=a.decompose(k,P)
        self.assertEqual(metric,eta);self.assertTrue(all(x==a.ZERO for x in residual+antisym))
        arbitrary=[[F(101+i+j) for j in range(3)] for i in range(3)]
        rows=a.contractions(k,[a.ZERO]*4,P,arbitrary,[F(0)]*3,[F(0)]*3,metric,b,'formal_supplied_S')
        self.assertEqual([r[0] for r in rows],[[a.scale(2*x,eta) for x in p] for p in P])

    def test_component_weights_no_offdiagonal_factor_two(self):
        values=[a.ZERO]*10;values[1]=c(3);k=a.tensor_from_components(values)
        self.assertEqual(k[0][1],c(3));self.assertEqual(k[1][0],c(3))
        rows=a.contractions(k,[a.ZERO]*4,[[F(2),F(0),F(0),F(0)],P[1],P[2]],[[F(0)]*3 for _ in range(3)],[F(0)]*3,[F(0)]*3,a.ZERO,[[a.ZERO]*3 for _ in range(3)],'exact_shift_cartesian')
        self.assertEqual(rows[0][0][1],c(12))

    def test_exact_geometry_formal_equals_cartesian_with_metric(self):
        metric=c(7,1);b=[[c(i+j+1,i-j) for j in range(3)] for i in range(3)]
        b=[[b[min(i,j)][max(i,j)] for j in range(3)] for i in range(3)]
        k=[[a.add(a.scale(a.METRIC[mu] if mu==nu else 0,metric),a.total(a.scale(P[i][mu]*P[j][nu],b[i][j]) for i in range(3) for j in range(3))) for nu in range(4)] for mu in range(4)]
        z=[[2*a.dot(x,y) for y in P] for x in P];jm=[c(2,1)]*4;f=[F(1),F(2),F(3)]
        self.assertEqual(a.contractions(k,jm,P,z,f,f,metric,b,'formal_supplied_S'),a.contractions(k,jm,P,z,f,f,metric,b,'exact_shift_cartesian'))
        recovered,bb,residual,_=a.decompose(k,P);self.assertEqual(recovered,metric);self.assertEqual(bb,b);self.assertEqual(residual,[a.ZERO]*10)

    def test_slot_selection_and_already_squared_mass(self):
        record={'raw':[['1','1','2','3'],['1','2','5','7'],['1','3','11','13'],['1','4','17','19']]}
        slots=a.raw_slots(record,F);self.assertEqual([slots(1,i) for i in range(3)],[c(17,19),c(7,10),c(11,13)])
        shifts={1:[F(0)]*4,2:[F(2),F(0),F(0),F(0)]};masses={1:c(9),2:c(25)};s={(1,1):F(0),(2,2):F(0),(1,2):F(4),(2,1):F(4)}
        self.assertEqual(a.geometry(shifts,s,masses)[4],[F(-12)])
        masses[1]=c(9,1)
        with self.assertRaises(ValueError):a.geometry(shifts,s,masses)

    def test_explicit_singular_refusal_and_zero_scale(self):
        with self.assertRaises(ValueError):a.solve([[1,1],[1,1]],[1,1])
        with self.assertRaises(ValueError):a.decompose([[a.ZERO]*4 for _ in range(4)],[P[0],P[0],P[0]])
        self.assertEqual(a.norm([a.ZERO]*4),0);self.assertIsNone(a.ratio(F(1),F(0)));self.assertIsNone(a.ratio(F(0),F(0)))

    def test_unmatched_and_swapped_pinch_reject(self):
        box={'scale':'1','masses':[[str(i),str(i), '0'] for i in range(1,5)],'invariants':[[str(i),str(j),str((i-j)**2)] for i in range(1,5) for j in range(1,5)]}
        triangle={'status':0,'rank':1,'n':3,'scale':'1','masses':box['masses'][:3],'invariants':[[str(i),str(j),str((i-j)**2)] for i in range(1,4) for j in range(1,4)]}
        a.bind_pinch(box,triangle,(1,2,3),F)
        with self.assertRaises(ValueError):a.bind_pinch(box,triangle,(2,1,3),F)
        bad=copy.deepcopy(triangle);bad['invariants'][1][2]='99'
        with self.assertRaises(ValueError):a.bind_pinch(box,bad,(1,2,3),F)


if __name__=='__main__':unittest.main()
