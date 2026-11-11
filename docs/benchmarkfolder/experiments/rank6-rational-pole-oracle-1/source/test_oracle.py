import unittest
from fractions import Fraction as F
from math import comb
from oracle import C,ZERO,bubble_uv,tadpole_uv,triangle_poles,scaleless_bubble_poles,monomials,ieee_fraction

class ExactPoleTests(unittest.TestCase):
    def test_ordering_and_counts(self):
        aa=monomials(6)
        self.assertEqual(len(aa),210)
        self.assertEqual(len(set(aa)),210)
        self.assertEqual(aa[55],(1,1,1,1))
        self.assertEqual(aa[161],(3,1,1,1))
        for r in range(7):self.assertEqual(sum(sum(a)==r for a in aa),comb(r+3,3))
    def test_scalar_and_odd_sign_controls(self):
        p=[(0,0,0,0),(1,3,2,4)];m=[(256,F(-1,8)),(264,F(-1,4))]
        self.assertEqual(bubble_uv((0,0,0,0),p,m,-28),C(1))
        for mu in range(4):
            a=tuple(int(i==mu) for i in range(4))
            self.assertEqual(bubble_uv(a,p,m,-28),C(F(p[1][mu],2)))
        self.assertEqual(bubble_uv((2,0,0,0),p,m,-28),C(F(398,3),F(-3,32)))
    def test_equal_mass_radial_bubble(self):
        p=[(0,0,0,0)]*2;m=[(4,0)]*2
        self.assertEqual(bubble_uv((2,0,0,0),p,m,0),C(2))
        self.assertEqual(bubble_uv((0,2,0,0),p,m,0),C(-2))
        self.assertEqual(bubble_uv((4,0,0,0),p,m,0),C(6))
        self.assertEqual(bubble_uv((2,2,0,0),p,m,0),C(-2))
        self.assertEqual(bubble_uv((6,0,0,0),p,m,0),C(20))
    def test_tadpole_angular_controls(self):
        p=(0,0,0,0);m=(4,0)
        self.assertEqual(tadpole_uv((0,0,0,0),p,m),C(4))
        self.assertEqual(tadpole_uv((2,0,0,0),p,m),C(4))
        self.assertEqual(tadpole_uv((0,2,0,0),p,m),C(-4))
        self.assertEqual(tadpole_uv((4,0,0,0),p,m),C(8))
        self.assertEqual(tadpole_uv((6,0,0,0),p,m),C(20))
    def test_scalar_triangle_and_all_nonzero_IR2(self):
        self.assertEqual(triangle_poles((0,0,0,0)),(ZERO,ZERO,C(F(-1,4))))
        for a in monomials(6):self.assertNotEqual(triangle_poles(a)[2],ZERO)
    def test_scaleless_all_nonzero_and_cancel(self):
        for a in monomials(6):
            uv,ir1,ir2=scaleless_bubble_poles(a)
            self.assertNotEqual(uv,ZERO)
            self.assertEqual(uv+ir1,ZERO)
            self.assertEqual(ir2,ZERO)
    def test_triangle_trace_all_poles_through_rank6(self):
        v=(3,1,2,2);w=(0,2,0,0)
        for a in monomials(4):
            trace=[ZERO]*3
            for mu in range(4):
                b=list(a);b[mu]+=2
                poles=triangle_poles(b)
                for j in range(3):trace[j]=trace[j]+poles[j]*(1 if mu==0 else -1)
            bubble=bubble_uv(a,[v,w],[(0,0)]*2,0)
            self.assertEqual(trace,[bubble,-bubble,ZERO])
    def test_triangle_linear_denominator_identities_through_rank6(self):
        v=(3,1,2,2);w=(0,2,0,0);origin=(0,0,0,0)
        for a in monomials(5):
            contracted=[]
            for p in [v,w]:
                value=[ZERO]*3
                for mu in range(4):
                    b=list(a);b[mu]+=1
                    terms=triangle_poles(b)
                    for j in range(3):value[j]=value[j]+terms[j]*(p[mu]*(1 if mu==0 else -1))
                contracted.append(value)
            bw=bubble_uv(a,[v,w],[(0,0)]*2,0)
            b0w=bubble_uv(a,[origin,w],[(0,0)]*2,-4)
            b0v=bubble_uv(a,[origin,v],[(0,0)]*2,0)
            t=triangle_poles(a)
            self.assertEqual(contracted[0],[(bw-b0w)/2,-bw/2,ZERO])
            self.assertEqual(contracted[1],[(bw-b0v-t[0]*4)/2,(-bw+b0v-t[1]*4)/2,-t[2]*2])
    def test_binary128_exact_reader(self):
        self.assertEqual(ieee_fraction(['0000000000000000','3FFF000000000000']),F(1))
        self.assertEqual(ieee_fraction(['0000000000000001','3FFF000000000000']),F(1)+F(1,2**112))
        self.assertEqual(ieee_fraction(['0000000000000000','8000000000000000']),F(0))
        with self.assertRaises(ValueError):ieee_fraction(['0000000000000000','7FFF000000000000'])

if __name__=='__main__':unittest.main()
