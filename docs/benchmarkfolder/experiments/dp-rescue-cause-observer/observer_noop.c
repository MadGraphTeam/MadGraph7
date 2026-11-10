/* Plain-control linkage only: original installed DSO, no private observer.
 * The common Fortran driver still records every public input/output/IEEE state.
 * This shim must never be linked into the observed or observer-disabled arm. */
#include <stdint.h>
#include <stddef.h>
int ml_tirust_observer_open(const uint8_t *p,size_t n,int enabled) {
    (void)p; (void)n; return enabled == 0 ? 0 : 1;
}
void ml_tirust_observer_begin_point(int64_t point,int phase) {(void)point;(void)phase;}
void ml_tirust_observer_event(int tag,const int64_t *words,size_t n) {(void)tag;(void)words;(void)n;}
int ml_tirust_observer_flush(void) {return 0;}
int ml_tirust_observer_close(void) {return 0;}
uint64_t ml_tirust_observer_health(void) {return 0;}
