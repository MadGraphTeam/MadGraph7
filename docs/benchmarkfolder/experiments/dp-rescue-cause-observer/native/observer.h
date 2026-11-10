/* Private experiment ABI, never an installed numerical interface. */
#ifndef MADLOOP_DP_RESCUE_OBSERVER_H
#define MADLOOP_DP_RESCUE_OBSERVER_H
#include <stddef.h>
#include <stdint.h>
/* Exclusive create; call once outside amplitudes. enabled must be 0 or 1. */
int ml_tirust_observer_open(const uint8_t *path, size_t length, int enabled);
void ml_tirust_observer_begin_point(int64_t point, int phase);
void ml_tirust_observer_event(int tag, const int64_t *words, size_t count);
/* Outside amplitudes only. Flush clears retained events, never health/counts. */
int ml_tirust_observer_flush(void);
int ml_tirust_observer_close(void);
uint64_t ml_tirust_observer_health(void);
#endif
