/* Source-only observer proposal. No wrapper performs I/O or allocation.
 * All buffer manipulation uses memcpy/integers; native calls forward once.
 * observer.txt JSONL schema 1: action, kind, metadata, input/output byte-hex,
 * native flags/rounding before/after, observer health. DP bytes are native
 * little-endian GNU/Linux; MP bytes are the actual 64-byte decimal records.
 * One native evaluation per action. Overflow or a second call fails closed.
 */
#include "tirust.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <fenv.h>

#ifndef ORIGINAL_NC
#error ORIGINAL_NC required
#endif
static FILE *stream;
static int health, action, present, n0, rank0, mode0, finite0, check0, cache0;
static int status0, flags0, flags1, flags2, round0, round1, round2, kind0;
static unsigned char p0[24*64], s0[36*64], m0[12*64], mu0[64], out0[1680*64];
static size_t pn, sn, mn, mun, on;
static unsigned char error0[2048];
static double captured_p[24], captured_m[12], captured_c[6*ORIGINAL_NC];
static int captured, captured_stable;
static unsigned char cp[40*8], cs[30*8], cm[12*8];
static size_t cpn, csn, cmn;

static void hex(const unsigned char *data, size_t count) {
  size_t i; fputc('"',stream);
  for(i=0;i<count;i++) fprintf(stream,"%02x",data[i]);
  fputc('"',stream);
}
void rp_open_(void) { stream=fopen("observer.txt","wx"); if(!stream) health=1; }
void rp_close_(void) { if(!stream || fclose(stream)) health=1; stream=NULL; }
int rp_health_(void) { return health; }
void rp_begin_(const int *id) {
  if(present) health=1;
  action=*id; present=0; kind0=0; cpn=csn=cmn=0;
  n0=rank0=mode0=finite0=check0=cache0=status0=0;
  flags0=flags1=flags2=round0=round1=round2=0;
}
void rp_state_(int *flags,int *rounding) { *flags=fetestexcept(FE_ALL_EXCEPT); *rounding=fegetround(); }
static void before(int kind,int n,int rank,int mode,int finite,int check,int cache) {
  if(present) health=1;
  present=1; kind0=kind; n0=n; rank0=rank; mode0=mode; finite0=finite; check0=check; cache0=cache;
  flags0=fetestexcept(FE_ALL_EXCEPT); round0=fegetround();
  memset(error0,0,sizeof(error0));
}
static void copy(unsigned char *dst,size_t cap,const void *src,size_t count) {
  if(count>cap || (!src && count)) {health=1;return;}
  memcpy(dst,src,count);
}
static void after(int status) {
  status0=status; flags1=fetestexcept(FE_ALL_EXCEPT); round1=fegetround();
  if(status!=TIRUST_SUCCESS) {
    size_t need=ml_tirust_last_error(error0,sizeof(error0));
    if(need>sizeof(error0)) health=1;
    error0[sizeof(error0)-1]=0;
  }
  flags2=fetestexcept(FE_ALL_EXCEPT); round2=fegetround();
  if(flags1!=flags2 || round1!=round2) health=1;
}
extern int32_t __real_ml_tirust_evaluate_dp(ml_tirust_context*,int32_t,int32_t,
 const double*,size_t,const double*,size_t,const ml_tirust_complex*,size_t,double,
 ml_tirust_complex*,size_t,int32_t,int32_t,int32_t,int32_t);
int32_t __wrap_ml_tirust_evaluate_dp(ml_tirust_context *ctx,int32_t n,int32_t rank,
 const double *p,size_t pl,const double *s,size_t sl,const ml_tirust_complex *m,size_t ml,double mu,
 ml_tirust_complex *out,size_t ol,int32_t mode,int32_t finite,int32_t check,int32_t cache) {
  int32_t status;
  before(1,n,rank,mode,finite,check,cache);
  pn=pl*8;sn=sl*8;mn=ml*16;mun=8;on=ol*16;
  copy(p0,sizeof(p0),p,pn);copy(s0,sizeof(s0),s,sn);copy(m0,sizeof(m0),m,mn);copy(mu0,sizeof(mu0),&mu,mun);
  if(flags0!=fetestexcept(FE_ALL_EXCEPT)||round0!=fegetround())health=1;
  status=__real_ml_tirust_evaluate_dp(ctx,n,rank,p,pl,s,sl,m,ml,mu,out,ol,mode,finite,check,cache);
  after(status);
  if(status==0)copy(out0,sizeof(out0),out,on);else on=0;
  if(flags2!=fetestexcept(FE_ALL_EXCEPT)||round2!=fegetround())health=1;
  return status;
}
extern int32_t __real_ml_tirust_evaluate_mp(ml_tirust_context*,int32_t,int32_t,
 const unsigned char*,size_t,const unsigned char*,size_t,const unsigned char*,size_t,
 const unsigned char*,size_t,unsigned char*,size_t,int32_t,int32_t,int32_t,int32_t);
int32_t __wrap_ml_tirust_evaluate_mp(ml_tirust_context *ctx,int32_t n,int32_t rank,
 const unsigned char *p,size_t pl,const unsigned char *s,size_t sl,const unsigned char *m,size_t ml,
 const unsigned char *mu,size_t mul,unsigned char *out,size_t ol,int32_t mode,int32_t finite,int32_t check,int32_t cache) {
  int32_t status;
  before(2,n,rank,mode,finite,check,cache);
  pn=pl;sn=sl;mn=ml;mun=mul;on=ol;
  copy(p0,sizeof(p0),p,pn);copy(s0,sizeof(s0),s,sn);copy(m0,sizeof(m0),m,mn);copy(mu0,sizeof(mu0),mu,mun);
  if(flags0!=fetestexcept(FE_ALL_EXCEPT)||round0!=fegetround())health=1;
  status=__real_ml_tirust_evaluate_mp(ctx,n,rank,p,pl,s,sl,m,ml,mu,mul,out,ol,mode,finite,check,cache);
  after(status);
  if(status==0)copy(out0,sizeof(out0),out,on);else on=0;
  if(flags2!=fetestexcept(FE_ALL_EXCEPT)||round2!=fegetround())health=1;
  return status;
}
extern void __real_ml5_0_tirustloop_(int*,int*,int*,double*,double*,double*,int*);
void __wrap_ml5_0_tirustloop_(int *ct,int *n,int *rank,double *p,double *m,double *coefs,int *stable) {
  int f=fetestexcept(FE_ALL_EXCEPT),r=fegetround();
  if(*n!=6 || (*rank!=4 && *rank!=5))health=1;
  memcpy(captured_p,p,sizeof(captured_p));memcpy(captured_m,m,sizeof(captured_m));
  if(f!=fetestexcept(FE_ALL_EXCEPT)||r!=fegetround())health=1;
  __real_ml5_0_tirustloop_(ct,n,rank,p,m,coefs,stable);
  f=fetestexcept(FE_ALL_EXCEPT);r=fegetround();
  memcpy(captured_c,coefs,sizeof(captured_c));captured_stable=*stable;captured++;
  if(f!=fetestexcept(FE_ALL_EXCEPT)||r!=fegetround())health=1;
}
void rp_capture_(double *p,double *m,double *coefs,int *stable,int *count) {
  memcpy(p,captured_p,sizeof(captured_p));memcpy(m,captured_m,sizeof(captured_m));
  memcpy(coefs,captured_c,sizeof(captured_c));*stable=captured_stable;*count=captured;
}
/* Called once after the private production COLLIER adapter built its actual
 * inputs, immediately before its first TNTEN call. No arithmetic or I/O. */
void rp_cll_input_(int *n,int *rank,void *p,void *s,void *m) {
  if(cpn || *n<2 || *n>6)health=1;
  n0=*n;rank0=*rank;cpn=(size_t)8*(*n-1)*8;csn=(size_t)(*n)*(*n-1)*8;cmn=(size_t)2*(*n)*8;
  copy(cp,sizeof(cp),p,cpn);copy(cs,sizeof(cs),s,csn);copy(cm,sizeof(cm),m,cmn);
}
void rp_flush_(void) {
  if(!stream){health=1;return;}
  fprintf(stream,"{\"schema\":1,\"action\":%d,\"present\":%d,\"kind\":%d,\"n\":%d,\"rank\":%d,\"mode\":%d,\"finite\":%d,\"check\":%d,\"cache\":%d,\"status\":%d,\"flags\":[%d,%d,%d],\"rounding\":[%d,%d,%d],\"health\":%d,\"p\":",action,present,kind0,n0,rank0,mode0,finite0,check0,cache0,status0,flags0,flags1,flags2,round0,round1,round2,health);
  /* On invalid lengths retain health failure without an out-of-bounds read. */
  hex(p0,present&&pn<=sizeof(p0)?pn:0);fprintf(stream,",\"s\":");hex(s0,present&&sn<=sizeof(s0)?sn:0);
  fprintf(stream,",\"m\":");hex(m0,present&&mn<=sizeof(m0)?mn:0);
  fprintf(stream,",\"mu\":");hex(mu0,present&&mun<=sizeof(mu0)?mun:0);fprintf(stream,",\"out\":");hex(out0,present&&on<=sizeof(out0)?on:0);
  fprintf(stream,",\"error\":");hex(error0,present?strlen((char*)error0):0);
  fprintf(stream,",\"cll_p\":");hex(cp,cpn<=sizeof(cp)?cpn:0);fprintf(stream,",\"cll_s\":");hex(cs,csn<=sizeof(cs)?csn:0);
  fprintf(stream,",\"cll_m\":");hex(cm,cmn<=sizeof(cm)?cmn:0);
  fprintf(stream,"}\n");if(fflush(stream))health=1;present=0;cpn=csn=cmn=0;
}
