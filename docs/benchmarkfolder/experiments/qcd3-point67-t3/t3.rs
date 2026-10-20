//! Diagnostic rank<=4, mindeg2 T3 arithmetic variants. Original-order variant
//! adapts Ninja's tensor_ninja.cc ten_build/nlr_ten/ten_contract (GPL).
use ninjars::fast::{C64,CMom};
use ninjars::fast::tensor_layout::{ncoeff,ncoeff_homogeneous,block_offset};
#[derive(Default,Clone,Copy)]
struct Sum {sum:f64,err:f64}
impl Sum {
 fn add(&mut self,x:f64){let s=self.sum+x;self.err+=if self.sum.abs()>=x.abs(){(self.sum-s)+x}else{(x-s)+self.sum};self.sum=s;}
 fn product(&mut self,a:f64,b:f64,sign:f64){let hi=a*b;let lo=a.mul_add(b,-hi);self.add(sign*hi);self.add(sign*lo);}
 fn value(self)->f64{self.sum+self.err}
}
#[derive(Default,Clone,Copy)]
struct Acc {r:Sum,i:Sum}
impl Acc {
 fn add(&mut self,z:C64){self.r.add(z.re);self.i.add(z.im);}
 fn product(&mut self,a:C64,b:C64,exact:bool){if exact{self.r.product(a.re,b.re,1.);self.r.product(a.im,b.im,-1.);self.i.product(a.re,b.im,1.);self.i.product(a.im,b.re,1.);}else{self.add(a*b);}}
 fn value(self)->C64{C64::new(self.r.value(),self.i.value())}
}
fn index(d:usize,i:usize,component:usize)->Option<usize>{match component{
 0=>(i==0).then_some(0),1=>(i<d).then_some(1+i),
 2=>(i<d*(d+1)/2).then_some(1+d+i),3=>Some(1+d+d*(d+1)/2+i),_=>unreachable!()}}
fn horner(rank:usize,c:&[C64],a:&CMom,e3:&CMom,e4:&CMom,p:C64,exact:bool)->[C64;4]{
 let mut g:Vec<[C64;4]>=c[block_offset(rank)..].iter().map(|&z|[z,C64::ZERO,C64::ZERO,C64::ZERO]).collect();
 for d in (1..=rank).rev(){
  let mut out=vec![[C64::ZERO;4];ncoeff_homogeneous(d-1)];
  for (i,row) in out.iter_mut().enumerate(){
   let mut sums=[Acc::default();4];
   for component in [3,2,1,0]{if let Some(j)=index(d,i,component){
    let v=g[j];
    sums[0].product(v[0],e3[component],exact);
    sums[1].product(v[1],e3[component],exact);sums[1].product(v[0],a[component],exact);
    sums[2].product(v[2],e3[component],exact);sums[2].product(v[1],a[component],exact);
    sums[3].product(v[3],e3[component],exact);sums[3].product(v[0],e4[component],exact);
   }}
   let power=rank-d+1;if power<=2{sums[power].add(c[block_offset(d-1)+i]);}
   *row=sums.map(Acc::value);
  }g=out;
 }
 let jet=g[0];let mut constant=Acc::default();constant.add(jet[2]);constant.product(p,jet[3],exact);
 [jet[0],jet[1],constant.value(),jet[3]]
}
fn build(d:usize,lower:&[C64],v:&CMom)->Vec<C64>{let mut out=vec![lower[0]*v[0]];for (component,count) in [(1,d),(2,d*(d+1)/2),(3,ncoeff_homogeneous(d-1))]{for &x in &lower[..count]{out.push(v[component]*x);}}out}
fn from_mom(rank:usize,v:&CMom)->Vec<C64>{let mut out=vec![C64::real(1.)];for d in 1..=rank{let next=build(d,&out[block_offset(d-1)..],v);out.extend(next);}out}
fn nlr(rank:usize,k:usize,lower:&[C64],v0:&CMom,v1:&CMom)->Vec<C64>{let mut out=vec![C64::ZERO;block_offset(k)];for d in k..=rank{let mut next=build(d,&lower[block_offset(d-1)..],v1);if d>k{let extra=build(d,&out[block_offset(d-1)..],v0);for(x,y)in next.iter_mut().zip(extra){*x+=y;}}out.extend(next);}assert_eq!(out.len(),ncoeff(rank));out}
fn contract(d:usize,c:&[C64],v:&[C64],mode:usize)->C64{let off=block_offset(d);let count=ncoeff_homogeneous(d);if mode==3{let mut sum=C64::ZERO;for(&x,&y)in c[off..off+count].iter().zip(&v[off..off+count]){sum+=x*y;}sum}else{let mut sum=Acc::default();for(&x,&y)in c[off..off+count].iter().zip(&v[off..off+count]){sum.product(x,y,mode==5);}sum.value()}}
fn original_order(rank:usize,c:&[C64],a:&CMom,e3:&CMom,e4:&CMom,p:C64,mode:usize)->[C64;4]{
 let t0=from_mom(rank,e3);let t1=nlr(rank,1,&t0,e3,a);let t2=nlr(rank,2,&t1,e3,a);let t3=nlr(rank,1,&t0,e3,e4);
 let c0=contract(rank,c,&t0,mode);let c1a=contract(rank,c,&t1,mode);let c1b=contract(rank-1,c,&t0,mode);
 let c3=contract(rank,c,&t3,mode);let c2a=contract(rank-2,c,&t0,mode);let c2b=contract(rank-1,c,&t1,mode);let c2c=contract(rank,c,&t2,mode);
 if mode==3{[c0,c1a+c1b,c2a+c2b+c2c+p*c3,c3]}else{let mut c1=Acc::default();c1.add(c1a);c1.add(c1b);let mut c2=Acc::default();c2.add(c2a);c2.add(c2b);c2.add(c2c);c2.product(p,c3,mode==5);[c0,c1.value(),c2.value(),c3]}
}
// Standard compensated Horner: carry the first-order residual through the
// next polynomial level instead of rounding every corrected level back to DP.
fn carried_product(out:&mut Acc,g:Acc,x:C64){
 out.product(C64::new(g.r.sum,g.i.sum),x,true);
 out.product(C64::new(g.r.err,g.i.err),x,false);
}
fn carried(rank:usize,c:&[C64],a:&CMom,e3:&CMom,e4:&CMom,p:C64)->[C64;4]{
 let mut g:Vec<[Acc;4]>=c[block_offset(rank)..].iter().map(|&z|{let mut r=[Acc::default();4];r[0].add(z);r}).collect();
 for d in (1..=rank).rev(){
  let mut out=vec![[Acc::default();4];ncoeff_homogeneous(d-1)];
  for (i,sums) in out.iter_mut().enumerate(){
   for component in [3,2,1,0]{if let Some(j)=index(d,i,component){let v=g[j];
    carried_product(&mut sums[0],v[0],e3[component]);
    carried_product(&mut sums[1],v[1],e3[component]);carried_product(&mut sums[1],v[0],a[component]);
    carried_product(&mut sums[2],v[2],e3[component]);carried_product(&mut sums[2],v[1],a[component]);
    carried_product(&mut sums[3],v[3],e3[component]);carried_product(&mut sums[3],v[0],e4[component]);
   }}
   let power=rank-d+1;if power<=2{sums[power].add(c[block_offset(d-1)+i]);}
  }g=out;
 }
 let jet=g[0];let mut constant=jet[2];carried_product(&mut constant,jet[3],p);
 [jet[0].value(),jet[1].value(),constant.value(),jet[3].value()]
}
pub fn evaluate(rank:usize,c:&[C64],a:&CMom,e3:&CMom,e4:&CMom,p:C64,mode:usize)->[C64;4]{assert!((2..=4).contains(&rank));match mode{6=>carried(rank,c,a,e3,e4,p),1|2=>horner(rank,c,a,e3,e4,p,mode==2),3..=5=>original_order(rank,c,a,e3,e4,p,mode),_=>panic!("unsupported mode")}}
