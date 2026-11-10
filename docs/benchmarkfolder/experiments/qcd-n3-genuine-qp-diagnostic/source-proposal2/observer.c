/* Private pass-through byte journal. No floating arithmetic, cache queries,
 * error getters, flag resets, or calls to another reduction are performed. */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define MAX_RECORDS 200000u
#define MAX_BYTES (128u * 1024u * 1024u)
struct record { int32_t call,kind,part,group,so,rank,mode; size_t start,size; };
static struct record *rows;
static unsigned char *bytes;
static size_t used,nrows;
static int32_t current=-1,observed=0,health=0;
extern void qg_state(int32_t *,int32_t *,int32_t *,int32_t *);
extern void qg_native_state(void);

int32_t qg_setup(int32_t observation) {
    if(rows || bytes || (observation!=0 && observation!=1)) return 1;
    rows=calloc(MAX_RECORDS,sizeof *rows); bytes=malloc(MAX_BYTES);
    observed=observation;
    if(!rows || !bytes) { health=1; return 1; }
    return 0;
}
void qg_call(int32_t call) { current=call; }
int32_t qg_enabled(void) { return observed && current>=100; }
int32_t qg_health(void) { return health; }
void qg_record(int32_t kind,int32_t part,const void *data,size_t size) {
    struct record *r;
    if(!rows || !bytes || nrows>=MAX_RECORDS || size>MAX_BYTES-used || (!data && size)) {
        health=1; return;
    }
    r=rows+nrows++; r->call=current; r->kind=kind; r->part=part;
    r->group=r->so=r->rank=r->mode=0;
    if(kind>=100) qg_state(&r->group,&r->so,&r->rank,&r->mode);
    r->start=used; r->size=size;
    if(size) memcpy(bytes+used,data,size);
    used+=size;
}
int32_t qg_flush(void) {
    FILE *f=fopen("trace.txt","wx");
    size_t i,j;
    if(!f) return 1;
    fprintf(f,"QG1 %d %zu %zu %d\n",observed,nrows,used,health);
    for(i=0;i<nrows;i++) {
        struct record *r=rows+i;
        fprintf(f,"E %zu %d %d %d %d %d %d %d %zu ",i+1,r->call,r->kind,r->part,
                r->group,r->so,r->rank,r->mode,r->size);
        for(j=0;j<r->size;j++) fprintf(f,"%02x",bytes[r->start+j]);
        fputc('\n',f);
    }
    fprintf(f,"COMPLETE\n");
    { int failed=ferror(f); if(fclose(f)) failed=1; if(failed) return 1; }
    return health;
}

static int active(void) { return qg_enabled(); }
static void rec(int k,int p,const void *a,size_t n) { if(active()) qg_record(k,p,a,n); }
static size_t count_coeff(int r) {
    if(r<0 || r>4) { health=1; return 0; }
    return (size_t)(r+1)*(r+2)*(r+3)*(r+4)/24;
}
static int valid(int n,int r) {
    if(n<1 || n>5 || r<0 || r>4) { health=1; return 0; }
    return 1;
}

/* CutTools' public ABI takes DP shifts/masses and complex DP outputs even
 * for modes4/5. Callback pointers are forwarded without substitution. */
typedef void (*numerator)(const void *,void *);
extern void __real_ctsxcut_(const int32_t*,const void*,const void*,const int32_t*,
    numerator,numerator,const int32_t*,const void*,const void*,void*,void*,void*,int32_t*);
void __wrap_ctsxcut_(const int32_t *mode,const void *roots,const void *mu,const int32_t *n,
    numerator dp,numerator qp,const int32_t *rank,const void *p,const void *m,
    void *out,void *cc,void *r1,int32_t *stable) {
    int take=active() && valid(*n,*rank);
    if(take) {
        qg_native_state(); rec(101,0,n,4); rec(101,1,rank,4); rec(101,2,mode,4);
        rec(101,3,p,32u*(size_t)*n); rec(101,4,m,16u*(size_t)*n);
        rec(101,5,mu,8); rec(101,6,roots,8);
    }
    __real_ctsxcut_(mode,roots,mu,n,dp,qp,rank,p,m,out,cc,r1,stable);
    if(take) {
        rec(101,9,stable,4);
        if(*stable) { rec(101,10,out,48); rec(101,11,cc,16); rec(101,12,r1,16); }
    }
}

/* Every opaque quad argument is copied as bytes; no C quad arithmetic. */
extern void __real_quadninja_tensor_evaluate_smat_cm_(const void*,const int32_t*,
    const int32_t*,const void*,const void*,const void*,const void*,void*,void*,int32_t*);
void __wrap_quadninja_tensor_evaluate_smat_cm_(const void *co,const int32_t *n,
    const int32_t *rank,const void *s,const void *p,const void *m,const void *mu,
    void *out,void *r1,int32_t *status) {
    int take=active() && valid(*n,*rank);
    if(take) {
        qg_native_state(); rec(106,0,n,4); rec(106,1,rank,4);
        rec(106,2,co,32*count_coeff(*rank)); rec(106,3,p,64u*(size_t)*n);
        rec(106,4,m,32u*(size_t)*n); rec(106,5,mu,16);
        rec(106,6,s,16u*(size_t)*n*(size_t)*n);
    }
    __real_quadninja_tensor_evaluate_smat_cm_(co,n,rank,s,p,m,mu,out,r1,status);
    if(take) {
        rec(106,9,status,4);
        if(*status==0) { rec(106,10,out,96); rec(106,11,r1,32); }
    }
}

extern int32_t __real_ninjars_evaluate_mp_with_poles(void*,int32_t,int32_t,size_t,
    const char*,size_t,const char*,size_t,const char*,size_t,const char*,size_t,
    const char*,size_t,char*,size_t,int32_t);
int32_t __wrap_ninjars_evaluate_mp_with_poles(void *ctx,int32_t n,int32_t rank,size_t count,
    const char *co,size_t clen,const char *s,size_t slen,const char *p,size_t plen,
    const char *m,size_t mlen,const char *mu,size_t mulen,char *out,size_t olen,int32_t pole) {
    int32_t status; int take=active() && valid(n,rank);
    if(take) {
        size_t expected=count_coeff(rank);
        if(count!=expected || clen!=2*64*count || plen!=4*64*(size_t)n ||
           slen!=64*(size_t)n*(size_t)n || mlen!=2*64*(size_t)n || mulen!=64 || olen!=8*64) health=1;
        qg_native_state(); rec(108,0,&n,4); rec(108,1,&rank,4);
        rec(108,2,co,clen); rec(108,3,p,plen); rec(108,4,m,mlen);
        rec(108,5,mu,mulen); rec(108,6,s,slen); rec(108,7,&pole,4);
    }
    status=__real_ninjars_evaluate_mp_with_poles(ctx,n,rank,count,co,clen,s,slen,p,plen,m,mlen,mu,mulen,out,olen,pole);
    if(take) { rec(108,9,&status,4); if(status==0) rec(108,10,out,olen); }
    return status;
}

extern int32_t __real_ml_tirust_evaluate_mp(void*,int32_t,int32_t,const char*,size_t,
    const char*,size_t,const char*,size_t,const char*,size_t,char*,size_t,int32_t,int32_t,int32_t,int32_t);
int32_t __wrap_ml_tirust_evaluate_mp(void *ctx,int32_t n,int32_t rank,const char *p,size_t plen,
    const char *s,size_t slen,const char *m,size_t mlen,const char *mu,size_t mulen,
    char *out,size_t olen,int32_t pole,int32_t finite,int32_t check,int32_t cache) {
    int32_t status,flags[4]={pole,finite,check,cache}; int take=active() && valid(n,rank);
    if(take) {
        if(plen!=4*64*(size_t)n || slen!=64*(size_t)n*(size_t)n ||
           mlen!=2*64*(size_t)n || mulen!=64 || olen!=8*64*count_coeff(rank)) health=1;
        qg_native_state(); rec(109,0,&n,4); rec(109,1,&rank,4);
        rec(109,3,p,plen); rec(109,4,m,mlen); rec(109,5,mu,mulen);
        rec(109,6,s,slen); rec(109,7,flags,16);
    }
    status=__real_ml_tirust_evaluate_mp(ctx,n,rank,p,plen,s,slen,m,mlen,mu,mulen,out,olen,pole,finite,check,cache);
    if(take) { rec(109,9,&status,4); if(status==0) rec(109,10,out,olen); }
    return status;
}
