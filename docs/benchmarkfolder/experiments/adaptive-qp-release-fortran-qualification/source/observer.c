/* Integer/memcpy-only forwarding observers. No library loading or arithmetic. */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <fenv.h>
#include <time.h>
#include <unistd.h>
static FILE *stream;
static int health, action, present, status0, n0, rank0, poles0, finite0, check0, cache0;
static int flags0,flags1,flags2,round0,round1,round2;
static size_t pn,sn,mn,un,on;
static unsigned char pbuf[1024],sbuf[1024],mbuf[512],ubuf[64],obuf[17920],errorbuf[512];
static struct { int status,flags_before,flags_after;size_t count;uint64_t words[11]; } snapshots[32];
static unsigned nsnapshots;
extern int32_t __real_ml_tirust_evaluate_mp(void*,int32_t,int32_t,
 const unsigned char*,size_t,const unsigned char*,size_t,const unsigned char*,size_t,
 const unsigned char*,size_t,unsigned char*,size_t,int32_t,int32_t,int32_t,int32_t);
extern size_t ml_tirust_last_error(unsigned char*,size_t);
extern int32_t __real_ml_tirust_private_qp_precision_snapshot(void*,uint64_t*,size_t);
void native_observer_open(void) {
 struct timespec delay={0,10000000};unsigned i;
 stream=fopen("observer.jsonl","wx");if(!stream)health=1;
 /* Keep even the tiny legacy worker alive until its actual ELF maps are
  * sampled. This bounded startup handshake is outside any tensor call. */
 for(i=0;i<1000 && access("maps-ready",F_OK);i++)nanosleep(&delay,NULL);
 if(access("maps-ready",F_OK))health=1;
}
void native_observer_begin(int id) {if(present)health=1;action=id;present=0;}
int native_observer_health(void) {return health;}
static void copy(unsigned char *out,size_t capacity,const void *in,size_t length) {
 if(length>capacity || (!in&&length)){health=1;return;}memcpy(out,in,length);
}
int32_t __wrap_ml_tirust_evaluate_mp(void *context,int32_t n,int32_t rank,
 const unsigned char *p,size_t plen,const unsigned char *s,size_t slen,const unsigned char *m,size_t mlen,
 const unsigned char *u,size_t ulen,unsigned char *out,size_t olen,int32_t poles,int32_t finite,int32_t check,int32_t cache) {
 if(present)health=1;
 present=1;n0=n;rank0=rank;poles0=poles;finite0=finite;check0=check;cache0=cache;
 flags0=fetestexcept(FE_ALL_EXCEPT);round0=fegetround();
 pn=plen;sn=slen;mn=mlen;un=ulen;on=olen;memset(errorbuf,0,sizeof(errorbuf));
 copy(pbuf,sizeof(pbuf),p,pn);copy(sbuf,sizeof(sbuf),s,sn);copy(mbuf,sizeof(mbuf),m,mn);copy(ubuf,sizeof(ubuf),u,un);
 if(flags0!=fetestexcept(FE_ALL_EXCEPT)||round0!=fegetround())health=1;
 status0=__real_ml_tirust_evaluate_mp(context,n,rank,p,plen,s,slen,m,mlen,u,ulen,out,olen,poles,finite,check,cache);
 flags1=fetestexcept(FE_ALL_EXCEPT);round1=fegetround();
 if(status0==0)copy(obuf,sizeof(obuf),out,on);
 else {on=0;ml_tirust_last_error(errorbuf,sizeof(errorbuf));errorbuf[511]=0;}
 flags2=fetestexcept(FE_ALL_EXCEPT);round2=fegetround();
 if(flags1!=flags2||round1!=round2)health=1;
 return status0;
}
int32_t __wrap_ml_tirust_private_qp_precision_snapshot(void *context,uint64_t *out,size_t count) {
 int flags=fetestexcept(FE_ALL_EXCEPT),rounding=fegetround();
 int32_t result=__real_ml_tirust_private_qp_precision_snapshot(context,out,count);
 int flags_after=fetestexcept(FE_ALL_EXCEPT),rounding_after=fegetround();
 if(nsnapshots>=32){health=1;return result;}
 snapshots[nsnapshots].status=result;snapshots[nsnapshots].count=count;
 snapshots[nsnapshots].flags_before=flags;snapshots[nsnapshots].flags_after=flags_after;
 if(result==0 && count==11)memcpy(snapshots[nsnapshots].words,out,88);
 if(flags_after!=fetestexcept(FE_ALL_EXCEPT)||rounding_after!=fegetround())health=1;
 /* The getter is allowed to touch its own flags. Only observer post-copy
  * noninterference is asserted; the raw before/after values remain recorded. */
 (void)rounding;nsnapshots++;return result;
}
static void hex(const unsigned char *p,size_t n,size_t cap) {
 size_t i;if(n>cap){health=1;n=0;}fputc('"',stream);for(i=0;i<n;i++)fprintf(stream,"%02x",p[i]);fputc('"',stream);
}
void native_observer_flush(void) {
 if(!stream){health=1;return;}
 fprintf(stream,"{\"action\":%d,\"present\":%d,\"status\":%d,\"n\":%d,\"rank\":%d,\"poles\":%d,\"finite\":%d,\"check\":%d,\"cache\":%d,\"health\":%d,\"flags\":[%d,%d,%d],\"rounding\":[%d,%d,%d],\"p\":",action,present,status0,n0,rank0,poles0,finite0,check0,cache0,health,flags0,flags1,flags2,round0,round1,round2);
 hex(pbuf,pn,sizeof(pbuf));fputs(",\"s\":",stream);hex(sbuf,sn,sizeof(sbuf));fputs(",\"m\":",stream);hex(mbuf,mn,sizeof(mbuf));
 fputs(",\"scale\":",stream);hex(ubuf,un,sizeof(ubuf));fputs(",\"output\":",stream);hex(obuf,on,sizeof(obuf));
 fputs(",\"error\":",stream);hex(errorbuf,strlen((char*)errorbuf),sizeof(errorbuf));fputs("}\n",stream);
 if(fflush(stream))health=1;
 present=0;
}
void native_observer_close(void) {
 unsigned i,j;FILE *raw;
 if(!stream||fclose(stream))health=1;
 stream=NULL;raw=fopen("snapshots.jsonl","wx");if(!raw){health=1;return;}
 for(i=0;i<nsnapshots;i++){
  fprintf(raw,"{\"ordinal\":%u,\"status\":%d,\"count\":%zu,\"flags\":[%d,%d],\"words\":[",i,
   snapshots[i].status,snapshots[i].count,snapshots[i].flags_before,snapshots[i].flags_after);
  for(j=0;j<11;j++)fprintf(raw,"%s\"%016llx\"",j?",":"",(unsigned long long)snapshots[i].words[j]);
  fputs("]}\n",raw);
 }
 if(fclose(raw))health=1;
}
