/* Test double only. No real bridge, integral evaluator or dlopen is linked. */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
static int live, mode, getters, creates, tensors;
static uint64_t values[11];
static char error[512];
int ml_tirust_context_create(int version, void **out) {
    if(version!=1) return 2;
    creates++; live=1; memset(values,0,sizeof(values)); error[0]=0; *out=&live; return 0;
}
int ml_tirust_context_destroy(void *context) { (void)context; live=0; error[0]=0; return 0; }
int ml_tirust_register_plan(void *context,int n,int rank) { (void)context;(void)n;(void)rank;error[0]=0;return 0; }
int ml_tirust_clear_cache(void *context) { (void)context;error[0]=0;return 0; }
int ml_tirust_set_cache_budget(void *context,int64_t bytes) { (void)context;(void)bytes;error[0]=0;return 0; }
int ml_tirust_supports(int n,int rank,int finite) { (void)n;(void)rank;(void)finite;return 0; }
int ml_tirust_cache_statistics(void *context,uint64_t *out,size_t n) { (void)context;memset(out,0,n*8);return 0; }
int ml_tirust_qp_cache_statistics(void *context,uint64_t *out,size_t n) { return ml_tirust_cache_statistics(context,out,n); }
int ml_tirust_evaluate_dp(void) { tensors++;return 2; }
int ml_tirust_evaluate_mp(void) { tensors++;return 2; }
size_t ml_tirust_last_error(unsigned char *out,size_t n) {
    size_t need=strlen(error)+1;
    if(out && n) { size_t count=need<n?need:n;memcpy(out,error,count);out[n-1]=0; }
    return need;
}
#ifndef OMIT_GETTER
int32_t ml_tirust_qp_precision_statistics(void *context,uint64_t *out,size_t count) {
    getters++; error[0]=0;
    if(context!=&live || !live || count!=11 || mode==1) {strcpy(error,"mock getter failure");return 2;}
    memcpy(out,values,(mode==2?10:11)*sizeof(uint64_t));return 0;
}
#endif
void mock_words(const uint64_t *words) {memcpy(values,words,sizeof(values));}
void mock_mode(int value) {mode=value;}
void mock_error(void) {strcpy(error,"retained numerical rejection");}
int mock_getters(void) {return getters;}
int mock_creates(void) {return creates;}
int mock_tensors(void) {return tensors;}
