#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <errno.h>
static uint64_t creates, clears, registrations, unused_clears, create_ns, clear_ns;
static int registered;
extern int __real_ml_tirust_context_create(unsigned, void **);
extern int __real_ml_tirust_clear_cache(void *);
extern int __real_ml_tirust_register_plan(void *, int, int);
static uint64_t now_ns(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value)) return 0;
    return (uint64_t)value.tv_sec * 1000000000ULL + value.tv_nsec;
}
int __wrap_ml_tirust_context_create(unsigned version, void **handle) {
    uint64_t before = now_ns();
    int result = __real_ml_tirust_context_create(version, handle), saved = errno;
    create_ns += now_ns() - before;
    ++creates;
    if (!result) registered = 0;
    errno = saved;
    return result;
}
int __wrap_ml_tirust_clear_cache(void *handle) {
    uint64_t before = now_ns();
    int result = __real_ml_tirust_clear_cache(handle), saved = errno;
    clear_ns += now_ns() - before;
    ++clears;
    if (!registered) ++unused_clears;
    errno = saved;
    return result;
}
int __wrap_ml_tirust_register_plan(void *handle, int n, int rank) {
    int result = __real_ml_tirust_register_plan(handle, n, rank);
    ++registrations;
    if (!result) registered = 1;
    return result;
}
__attribute__((destructor)) static void record(void) {
    FILE *stream = fopen("ffi-profile.json", "wx");
    if (!stream) return;
    fprintf(stream, "{\"create_calls\":%llu,\"clear_calls\":%llu,"
        "\"register_calls\":%llu,\"clears_before_first_plan_registration\":%llu,"
        "\"create_nanoseconds\":%llu,\"clear_nanoseconds\":%llu}\n",
        (unsigned long long)creates, (unsigned long long)clears,
        (unsigned long long)registrations, (unsigned long long)unused_clears,
        (unsigned long long)create_ns, (unsigned long long)clear_ns);
    fclose(stream);
}
