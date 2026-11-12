#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef struct { double re, im; } C64;
_Static_assert(sizeof(int)==4 && sizeof(size_t)==8 && sizeof(C64)==16,"qualified x86_64 ABI required");
static int trace_fd=-2;
static uint64_t sequence=0;
static uint64_t nanos(clockid_t clock) { struct timespec t; if (clock_gettime(clock,&t)) _exit(121); return (uint64_t)t.tv_sec*1000000000u+(uint64_t)t.tv_nsec; }
static void capture_maps(const char *trace_path) {
    char path[PATH_MAX];int length=snprintf(path,sizeof(path),"%s.maps",trace_path);
    if(length<=0||(size_t)length>=sizeof(path)) _exit(126);
    int in=open("/proc/self/maps",O_RDONLY|O_CLOEXEC),out=open(path,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
    if(in<0||out<0) _exit(126);
    char buffer[16384];ssize_t count;
    while((count=read(in,buffer,sizeof(buffer)))>0) {
      ssize_t done=0;while(done<count) { ssize_t written=write(out,buffer+done,(size_t)(count-done));if(written<=0) _exit(126);done+=written; }
    }
    if(count<0) _exit(126);
    if(close(in)||close(out)) _exit(126);
}
static void emit(const char *line, size_t length) { if (trace_fd==-2) { const char *path=getenv("ML_TIRUST_PROFILE"); trace_fd=path?open(path,O_CREAT|O_EXCL|O_WRONLY|O_CLOEXEC,0600):-1; if(trace_fd<0) _exit(122); capture_maps(path); } size_t done=0; while(done<length) { ssize_t n=write(trace_fd,line+done,length-done); if(n<=0) _exit(123); done+=(size_t)n; } }
static void *resolve(const char *name) { void *result=dlsym(RTLD_NEXT,name); if(!result) { static const char error[]="TIRust profiling shim: symbol resolution failed\n"; (void)write(STDERR_FILENO,error,sizeof(error)-1); _exit(124); } return result; }
typedef struct { uint64_t sequence,wall,cpu; } Stamp;
typedef unsigned __int128 U128;
static void hash_bytes(U128 *h,const void *ptr,size_t size) {
    uint64_t length=(uint64_t)size;
    const unsigned char *p=(const unsigned char *)&length;
    const U128 prime=((U128)UINT64_C(0x0000000001000000)<<64)|UINT64_C(0x000000000000013b);
    for(size_t i=0;i<sizeof(length);++i) *h=(*h^p[i])*prime;
    if(!ptr) return;
    p=ptr;for(size_t i=0;i<size;++i) *h=(*h^p[i])*prime;
}
static void fingerprint(char out[33],int n,int rank,int poles,int finite,int check,int cache,const void *p,size_t pl,const void *s,size_t sl,const void *m,size_t ml,const void *scale,size_t scale_len,size_t out_len) {
    U128 h=((U128)UINT64_C(0x6c62272e07bb0142)<<64)|UINT64_C(0x62b821756295c58d);
    const int mode[]={n,rank,poles,finite,check,cache};
    hash_bytes(&h,mode,sizeof(mode));hash_bytes(&h,&out_len,sizeof(out_len));
    hash_bytes(&h,p,pl);hash_bytes(&h,s,sl);hash_bytes(&h,m,ml);hash_bytes(&h,scale,scale_len);
    (void)snprintf(out,33,"%016" PRIx64 "%016" PRIx64,(uint64_t)(h>>64),(uint64_t)h);
}
static Stamp begin(const char *op,int n,int rank,int poles,int finite,int check,int cache,const char *fingerprint_value) { Stamp s={++sequence,nanos(CLOCK_MONOTONIC),nanos(CLOCK_THREAD_CPUTIME_ID)}; char line[512]; int length=snprintf(line,sizeof(line),"{\"event\":\"begin\",\"pid\":%ld,\"sequence\":%" PRIu64 ",\"operation\":\"%s\",\"n\":%d,\"rank\":%d,\"poles\":%d,\"finite\":%d,\"check\":%d,\"cache\":%d,\"input_fingerprint_fnv128\":\"%s\",\"monotonic_ns\":%" PRIu64 ",\"thread_cpu_ns\":%" PRIu64 "}\n",(long)getpid(),s.sequence,op,n,rank,poles,finite,check,cache,fingerprint_value?fingerprint_value:"",s.wall,s.cpu); if(length<=0||(size_t)length>=sizeof(line)) _exit(125); emit(line,(size_t)length); return s; }
static int end(Stamp s,int status) { uint64_t cpu=nanos(CLOCK_THREAD_CPUTIME_ID),wall=nanos(CLOCK_MONOTONIC); char line[320]; int length=snprintf(line,sizeof(line),"{\"event\":\"end\",\"pid\":%ld,\"sequence\":%" PRIu64 ",\"status\":%d,\"monotonic_ns\":%" PRIu64 ",\"wall_ns\":%" PRIu64 ",\"thread_cpu_ns\":%" PRIu64 "}\n",(long)getpid(),s.sequence,status,wall,wall-s.wall,cpu-s.cpu); if(length<=0||(size_t)length>=sizeof(line)) _exit(125); emit(line,(size_t)length); return status; }
static void first_failed_dp(Stamp stamp,int status,int n,int rank,const double *p,size_t pl,const double *s,size_t sl,const C64 *m,size_t ml,double scale,size_t out_len,int poles,int finite,int check,int cache,const char *fingerprint_value) {
    static unsigned char retained[7][7][2];static int fd=-1;
    if(status==0||n<1||n>6||rank<0||rank>6||check<0||check>1||retained[n][rank][check]) return;
    retained[n][rank][check]=1;
    if(fd<0) { char path[PATH_MAX];const char *trace_path=getenv("ML_TIRUST_PROFILE");int size=snprintf(path,sizeof(path),"%s.failed-dp.jsonl",trace_path?trace_path:"");if(size<=0||(size_t)size>=sizeof(path)) _exit(127);fd=open(path,O_CREAT|O_EXCL|O_WRONLY|O_CLOEXEC,0600);if(fd<0) _exit(127); }
    char line[8192];size_t used=0;
#define ADD(...) do { int size=snprintf(line+used,sizeof(line)-used,__VA_ARGS__);if(size<=0||(size_t)size>=sizeof(line)-used) _exit(127);used+=(size_t)size; } while(0)
    ADD("{\"sequence\":%" PRIu64 ",\"status\":%d,\"n\":%d,\"rank\":%d,\"poles\":%d,\"finite\":%d,\"check\":%d,\"cache\":%d,\"out_len\":%zu,\"input_fingerprint_fnv128\":\"%s\"",stamp.sequence,status,n,rank,poles,finite,check,cache,out_len,fingerprint_value);
    const void *arrays[]={p,s,m,&scale};const size_t lengths[]={pl,sl,ml*2,1};const char *names[]={"p_f64_bits","s_f64_bits","m_f64_bits","mu2_f64_bits"};
    for(size_t array=0;array<4;++array) {
      ADD(",\"%s\":[",names[array]);
      for(size_t i=0;i<lengths[array];++i) { uint64_t bits;memcpy(&bits,(const unsigned char *)arrays[array]+i*sizeof(double),sizeof(bits));ADD("%s\"%016" PRIx64 "\"",i?",":"",bits); }
      ADD("]");
    }
    ADD("}\n");size_t done=0;while(done<used) { ssize_t written=write(fd,line+done,used-done);if(written<=0) _exit(127);done+=(size_t)written; }
#undef ADD
}
#define LOAD(name,type) static type next=NULL; if(!next) *(void **)(&next)=resolve(#name)
int ml_tirust_context_create(uint32_t version,void **out) { typedef int(*Fn)(uint32_t,void **); LOAD(ml_tirust_context_create,Fn); Stamp t=begin("create",0,0,0,0,0,0,NULL); return end(t,next(version,out)); }
int ml_tirust_context_destroy(void *handle) { typedef int(*Fn)(void *); LOAD(ml_tirust_context_destroy,Fn); Stamp t=begin("destroy",0,0,0,0,0,0,NULL); return end(t,next(handle)); }
int ml_tirust_register_plan(void *handle,int n,int rank) { typedef int(*Fn)(void *,int,int); LOAD(ml_tirust_register_plan,Fn); Stamp t=begin("register",n,rank,0,0,0,0,NULL); return end(t,next(handle,n,rank)); }
int ml_tirust_clear_cache(void *handle) { typedef int(*Fn)(void *); LOAD(ml_tirust_clear_cache,Fn); Stamp t=begin("clear",0,0,0,0,0,0,NULL); return end(t,next(handle)); }
int ml_tirust_set_cache_budget(void *handle,int64_t bytes) { typedef int(*Fn)(void *,int64_t); LOAD(ml_tirust_set_cache_budget,Fn); Stamp t=begin("budget",0,0,0,0,0,0,NULL); return end(t,next(handle,bytes)); }
int ml_tirust_evaluate_dp(void *handle,int n,int rank,const double *p,size_t pl,const double *s,size_t sl,const C64 *m,size_t ml,double scale,C64 *out,size_t ol,int poles,int finite,int check,int cache) { typedef int(*Fn)(void *,int,int,const double *,size_t,const double *,size_t,const C64 *,size_t,double,C64 *,size_t,int,int,int,int); LOAD(ml_tirust_evaluate_dp,Fn); char hash[33]; fingerprint(hash,n,rank,poles,finite,check,cache,p,pl*sizeof(*p),s,sl*sizeof(*s),m,ml*sizeof(*m),&scale,sizeof(scale),ol); Stamp t=begin("dp",n,rank,poles,finite,check,cache,hash); int status=next(handle,n,rank,p,pl,s,sl,m,ml,scale,out,ol,poles,finite,check,cache); int result=end(t,status); first_failed_dp(t,status,n,rank,p,pl,s,sl,m,ml,scale,ol,poles,finite,check,cache,hash); return result; }
int ml_tirust_evaluate_mp(void *handle,int n,int rank,const unsigned char *p,size_t pl,const unsigned char *s,size_t sl,const unsigned char *m,size_t ml,const unsigned char *scale,size_t scale_len,unsigned char *out,size_t ol,int poles,int finite,int check,int cache) { typedef int(*Fn)(void *,int,int,const unsigned char *,size_t,const unsigned char *,size_t,const unsigned char *,size_t,const unsigned char *,size_t,unsigned char *,size_t,int,int,int,int); LOAD(ml_tirust_evaluate_mp,Fn); char hash[33]; fingerprint(hash,n,rank,poles,finite,check,cache,p,pl,s,sl,m,ml,scale,scale_len,ol); Stamp t=begin("qp",n,rank,poles,finite,check,cache,hash); return end(t,next(handle,n,rank,p,pl,s,sl,m,ml,scale,scale_len,out,ol,poles,finite,check,cache)); }
