/* Private diagnostic observer. Integer/byte operations only; no flag resets. */
#include "tirust.h"
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(double) == 8, "Requires the qualified binary64 ABI");
_Static_assert(FLT_RADIX == 2 && DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024,
               "Requires qualified IEEE binary64");
_Static_assert(sizeof(ml_tirust_complex) == 16, "Complex layout differs");
_Static_assert(offsetof(ml_tirust_complex,re)==0 && offsetof(ml_tirust_complex,im)==8,
               "Complex component offsets differ");
enum { MAX_P = 24, MAX_S = 36, MAX_M = 6, MAX_OUT = 504, ERR_CAP = 4096 };
static FILE *events;
static int case_id, phase, call_id, point, method, mode, group, sqso, nprop, rank;
static int selected, full_request, active, seen[3][2], unique_count;
static unsigned long long forwarded;

static void die(const char *message) { fprintf(stderr,"TCAP_IO_FAILURE %s\n",message); exit(90); }
static void flush_events(void) { if (fflush(events) || ferror(events)) die("event write"); }
static uint64_t word(const void *p) { uint64_t value; memcpy(&value,p,8); return value; }
static void json_bytes(FILE *f,const unsigned char *s,size_t length) {
    fputc('"',f);
    for(size_t i=0;i<length;i++) {
        unsigned c=s[i];
        if(c=='"'||c=='\\') { fputc('\\',f);fputc((int)c,f); }
        else if(c<32||c>=127) fprintf(f,"\\u%04x",c);
        else fputc((int)c,f);
    }
    fputc('"',f);
}
static void identity(FILE *f,const char *type) {
    fprintf(f,"{\"record_type\":\"%s\",\"case_id\":%d,\"phase\":%d,\"call\":%d,\"point\":%d,\"method\":%d,\"mode\":%d,\"group\":%d,\"sqso\":%d,\"n\":%d,\"rank\":%d",
        type,case_id,phase,call_id,point,method,mode,group,sqso,nprop,rank);
}
static void hex_array(FILE *f,const void *p,size_t words) {
    const unsigned char *bytes=p;fputc('[',f);
    for(size_t i=0;i<words;i++) fprintf(f,"%s\"%016" PRIx64 "\"",i?",":"",word(bytes+8*i));
    fputc(']',f);
}
void ml_tcap_init(int32_t which) {
    if(events || (which!=1 && which!=2)) die("invalid init");
    case_id=which; events=fopen("tirust-capture-events.jsonl","wx");
    if(!events) die("exclusive event open");
    fprintf(events,"{\"record_type\":\"init\",\"case_id\":%d,\"schema\":1,\"input_limit_per_case\":6,\"tensor_output_read_policy\":\"status0_only\"}\n",case_id);
    flush_events();
}
void ml_tcap_point(int32_t ph,int32_t call,int32_t p,int32_t meth,int32_t md) {
    if(!events || active) die("point lifecycle");
    phase=ph;call_id=call;point=p;method=meth;mode=md;selected=0;full_request=0;
}
void ml_tcap_group(int32_t order,int32_t g,int32_t md,int32_t n,int32_t r,
                   const double *pl,const ml_tirust_complex *masses) {
    if(!events || active) die("group lifecycle");
    sqso=order;group=g;mode=md;nprop=n;rank=r;
    int target=case_id==1?25:2;
    int matched=case_id==1?(g==59||g==61):(g==408||g==422);
    full_request=matched && phase==1 && point==target && method>=1 && method<=3;
    selected=full_request || (matched && case_id==2 && phase==0 && point==2);
    active=1;
    if(!selected) return;
    identity(events,"group_enter");
    int valid=n==(case_id==1?5:6) && r==(case_id==1?4:5) && pl && masses;
    fprintf(events,",\"entry_buffers_valid\":%s",valid?"true":"false");
    if(valid) { fputs(",\"pl_bits\":",events);hex_array(events,pl,4*(size_t)n);
        fputs(",\"mass_bits\":",events);hex_array(events,masses,2*(size_t)n); }
    fputs("}\n",events);flush_events();
}
void ml_tcap_stage(int32_t stage,int32_t detail) {
    if(!selected || !active) return;
    identity(events,"fortran_stage");fprintf(events,",\"stage\":%d,\"detail\":%d}\n",stage,detail);flush_events();
}
void ml_tcap_status(int32_t stage,int32_t status,const unsigned char *message,int32_t length) {
    if(!selected || !active) return;
    if(length<0 || length>512 || !message) die("Fortran error text shape");
    identity(events,"fortran_status");fprintf(events,",\"stage\":%d,\"status\":%d,\"error_encoding\":\"latin1-bytes\",\"error_required\":null,\"error_truncation\":\"unobserved\",\"error_buffer_bytes\":512,\"error\":",stage,status);
    json_bytes(events,message,(size_t)length);fputs("}\n",events);flush_events();
}
void ml_tcap_end(int32_t stable) {
    /* A tensor-cache hit may bypass the dispatch hook. Record no invented call. */
    if(!active) return;
    if(selected) { identity(events,"group_end");fprintf(events,",\"stable\":%s}\n",stable?"true":"false");flush_events(); }
    selected=0;full_request=0;active=0;
}
static void save_input(int n,int r,const double *p,size_t plen,const double *s,size_t slen,
        const ml_tirust_complex *m,size_t mlen,double mu2,size_t olen,int poles,int finite,int check,int cache) {
    int slot=case_id==1?(group==59?0:1):(group==408?0:1);
    if(unique_count>=6 || seen[method-1][slot]) die("duplicate or excessive full request");
    seen[method-1][slot]=1;unique_count++;
    char name[128];snprintf(name,sizeof name,"tirust-input-p%03d-m%d-g%03d.dat",point,method,group);
    FILE *f=fopen(name,"wx");if(!f) die("exclusive input open");
    fprintf(f,"MLTCAP_INPUT_V1\n%d %d %d %d %d %d %d %zu %zu %zu %zu %d %d %d %d\n%016" PRIx64 "\n",
      case_id,phase,point,method,group,n,r,plen,slen,mlen,olen,poles,finite,check,cache,word(&mu2));
    for(size_t i=0;i<plen;i++) fprintf(f,"%016" PRIx64 "\n",word(p+i));
    for(size_t i=0;i<slen;i++) fprintf(f,"%016" PRIx64 "\n",word(s+i));
    const unsigned char *mass_bytes=(const unsigned char *)m;
    for(size_t i=0;i<2*mlen;i++) fprintf(f,"%016" PRIx64 "\n",word(mass_bytes+8*i));
    if(fflush(f) || ferror(f) || fclose(f)) die("input write");
}

extern int32_t __real_ml_tirust_evaluate_dp(ml_tirust_context *,int32_t,int32_t,
    const double *,size_t,const double *,size_t,const ml_tirust_complex *,size_t,double,
    ml_tirust_complex *,size_t,int32_t,int32_t,int32_t,int32_t);
int32_t __wrap_ml_tirust_evaluate_dp(ml_tirust_context *ctx,int32_t n,int32_t r,
    const double *p,size_t plen,const double *s,size_t slen,const ml_tirust_complex *m,size_t mlen,double mu2,
    ml_tirust_complex *out,size_t olen,int32_t poles,int32_t finite,int32_t check,int32_t cache) {
    /* Snapshot only validated input buffers before forwarding. No FP operations. */
    double pp[MAX_P],ss[MAX_S];ml_tirust_complex mm[MAX_M];
    int watched=selected && active;
    size_t nc=0;
    if(r>=0 && r<=6) nc=(size_t)(r+1)*(size_t)(r+2)*(size_t)(r+3)*(size_t)(r+4)/24;
    int shape=watched && n==nprop && r==rank && n==(case_id==1?5:6) && r==(case_id==1?4:5)
      && plen==4*(size_t)n && slen==(size_t)n*n && mlen==(size_t)n && olen==4*nc
      && plen<=MAX_P && slen<=MAX_S && mlen<=MAX_M && olen<=MAX_OUT && p && s && m && out;
    if(shape) { memcpy(pp,p,plen*8);memcpy(ss,s,slen*8);memcpy(mm,m,mlen*sizeof *m); }
    forwarded++;
    int32_t status=__real_ml_tirust_evaluate_dp(ctx,n,r,p,plen,s,slen,m,mlen,mu2,out,olen,poles,finite,check,cache);
    /* No bridge helper is allowed between the real call and this error copy. */
    int saved_errno=errno;
    if(watched) {
        unsigned char error[ERR_CAP]={0};
        size_t required=ml_tirust_last_error(error,sizeof error);
        size_t length=0;while(length<sizeof error-1 && error[length]) length++;
        identity(events,"c_boundary");
        fprintf(events,",\"status\":%d,\"forwarded_ordinal\":%llu,\"input_shape_valid\":%s,\"full_request\":%s,\"error_required\":%zu,\"error_truncated\":%s,\"error_encoding\":\"latin1-bytes\",\"error\":",
            status,forwarded,shape?"true":"false",full_request?"true":"false",required,required>sizeof error?"true":"false");
        json_bytes(events,error,length);
        if(status==TIRUST_SUCCESS && shape) { fputs(",\"output_bits\":",events);hex_array(events,out,2*olen); }
        fputs("}\n",events);flush_events();
        if(full_request && shape) save_input(n,r,pp,plen,ss,slen,mm,mlen,mu2,olen,poles,finite,check,cache);
    }
    errno=saved_errno;
    return status;
}
