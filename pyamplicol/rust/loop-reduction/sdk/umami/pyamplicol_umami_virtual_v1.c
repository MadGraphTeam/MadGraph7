/* SPDX-License-Identifier: 0BSD */
#include "pyamplicol_umami_virtual_v1.h"
#include <stdint.h>
#include <stdlib.h>

/* Local diagnostics are static; NULL selects the SDK's thread-local diagnostic. */
static _Thread_local const char *local_error = "";
static UmamiStatus fail(UmamiStatus status, const char *message) {
    local_error = message;
    return status;
}
static UmamiStatus native_failure(void) {
    local_error = NULL;
    return UMAMI_ERROR;
}
const char *pyamplicol_umami_virtual_v1_last_error_message(void) {
    return local_error ? local_error : rusticol_virtual_last_error_message();
}
uint32_t pyamplicol_umami_virtual_v1_version(void) { return PYAMPLICOL_UMAMI_VIRTUAL_VERSION; }
UmamiStatus pyamplicol_umami_virtual_v1_supported_inputs(const int **keys, size_t *count) {
    static const int supported[] = {UMAMI_IN_MOMENTA, PYAMPLICOL_UMAMI_V1_IN_MU_R2,
        PYAMPLICOL_UMAMI_V1_IN_MU_R2_SCALAR, PYAMPLICOL_UMAMI_V1_IN_POINT_ID};
    if (!keys || !count) return fail(UMAMI_ERROR, "null supported-input query output");
    *keys = supported; *count = sizeof(supported)/sizeof(supported[0]);
    return UMAMI_SUCCESS;
}
UmamiStatus pyamplicol_umami_virtual_v1_supported_outputs(const int **keys, size_t *count) {
    static const int supported[] = {PYAMPLICOL_UMAMI_V1_OUT_BORN,
        PYAMPLICOL_UMAMI_V1_OUT_VIRTUAL, PYAMPLICOL_UMAMI_V1_OUT_STATUS};
    if (!keys || !count) return fail(UMAMI_ERROR, "null supported-output query output");
    *keys = supported; *count = sizeof(supported)/sizeof(supported[0]);
    return UMAMI_SUCCESS;
}
static int product(size_t a, size_t b, size_t *out) {
    if (a && b > SIZE_MAX/a) return 0;
    *out = a*b; return 1;
}
static int buffer_valid(const void *p, size_t capacity, size_t required,
                        size_t element_size, size_t alignment) {
    size_t bytes;
    return capacity >= required && product(required,element_size,&bytes) &&
        bytes <= PTRDIFF_MAX && (!required || (p && (uintptr_t)p % alignment == 0));
}
static void *array(size_t count, size_t element_size) {
    size_t bytes;
    if (!product(count,element_size,&bytes) || bytes > PTRDIFF_MAX) return NULL;
    return bytes ? malloc(bytes) : NULL;
}
static int successful(uint32_t status) {
    return status == RUSTICOL_VIRTUAL_SUCCESS || status == RUSTICOL_VIRTUAL_VERIFIED ||
        status == RUSTICOL_VIRTUAL_RESCUED;
}

static UmamiStatus evaluate_bulk(int correlated,
    rusticol_virtual_handle *handle, size_t count, size_t stride, size_t offset,
    size_t input_count, const int *input_keys, const void *const *inputs, const size_t *input_capacities,
    size_t output_count, const int *output_keys, void *const *outputs, const size_t *output_capacities,
    const pyamplicol_umami_virtual_v1_options *options, void **diagnostic_result) {
    if (diagnostic_result) *diagnostic_result = NULL;
    if (offset > stride || count > stride-offset)
        return fail(UMAMI_ERROR, "offset/count exceeds the UMAMI stride");
    if ((input_count && (!input_keys || !inputs || !input_capacities)) ||
        (output_count && (!output_keys || !outputs || !output_capacities)))
        return fail(UMAMI_ERROR, "missing UMAMI key/pointer/capacity arrays");
    const void *in[4] = {NULL,NULL,NULL,NULL};
    void *out[3] = {NULL,NULL,NULL};
    size_t in_capacity[4] = {0,0,0,0}, out_capacity[3] = {0,0,0};
    unsigned in_present = 0, out_present = 0;
    for (size_t i=0;i<input_count;i++) {
        unsigned slot;
        switch (input_keys[i]) {
        case UMAMI_IN_MOMENTA: slot=0; break;
        case PYAMPLICOL_UMAMI_V1_IN_MU_R2: slot=1; break;
        case PYAMPLICOL_UMAMI_V1_IN_MU_R2_SCALAR: slot=2; break;
        case PYAMPLICOL_UMAMI_V1_IN_POINT_ID: slot=3; break;
        default: return fail(UMAMI_ERROR_UNSUPPORTED_INPUT, "unsupported virtual-extension input key");
        }
        if (in_present & (1u<<slot)) return fail(UMAMI_ERROR, "duplicate virtual-extension input key");
        in_present |= 1u<<slot; in[slot]=inputs[i]; in_capacity[slot]=input_capacities[i];
    }
    for (size_t i=0;i<output_count;i++) {
        unsigned slot;
        switch (output_keys[i]) {
        case PYAMPLICOL_UMAMI_V1_OUT_BORN:
            if (correlated) return fail(UMAMI_ERROR_UNSUPPORTED_OUTPUT,"correlations require complex output keys");
            slot=0; break;
        case PYAMPLICOL_UMAMI_V1_OUT_VIRTUAL:
            if (correlated) return fail(UMAMI_ERROR_UNSUPPORTED_OUTPUT,"correlations require complex output keys");
            slot=1; break;
        case PYAMPLICOL_UMAMI_V1_OUT_CORRELATED_BORN:
            if (!correlated) return fail(UMAMI_ERROR_UNSUPPORTED_OUTPUT,"ordinary output uses real keys");
            slot=0; break;
        case PYAMPLICOL_UMAMI_V1_OUT_CORRELATED_VIRTUAL:
            if (!correlated) return fail(UMAMI_ERROR_UNSUPPORTED_OUTPUT,"ordinary output uses real keys");
            slot=1; break;
        case PYAMPLICOL_UMAMI_V1_OUT_STATUS: slot=2; break;
        default: return fail(UMAMI_ERROR_UNSUPPORTED_OUTPUT, "unsupported virtual-extension output key");
        }
        if (out_present & (1u<<slot)) return fail(UMAMI_ERROR, "duplicate virtual-extension output key");
        out_present |= 1u<<slot; out[slot]=outputs[i]; out_capacity[slot]=output_capacities[i];
    }
    if ((in_present & 6u) == 6u) return fail(UMAMI_ERROR, "provide one scale input, not both scalar and per-event");
    if (count && (!(in_present & 1u) || !(in_present & 6u)))
        return fail(UMAMI_ERROR_MISSING_INPUT, "momenta and renormalization scale squared are required");
    if (count && !(out_present & 4u))
        return fail(UMAMI_ERROR, "per-input status output is required for a nonempty virtual batch");
    uint32_t resolved=options ? options->resolved : 0;
    if (correlated && resolved) return fail(UMAMI_ERROR_NOT_IMPLEMENTED,"resolved directed correlations are unavailable");
    if (resolved > 1) return fail(UMAMI_ERROR, "resolved must be zero or one");
    rusticol_virtual_shape_info shape;
    if (rusticol_virtual_shape(handle,&shape)) return native_failure();
    if (resolved && !shape.resolved_available)
        return fail(UMAMI_ERROR_NOT_IMPLEMENTED, "saved runtime has no resolved virtual capability");
    rusticol_virtual_correlated_shape_info correlated_shape={0};
    if (correlated) {
        if (rusticol_virtual_correlated_shape(handle,&correlated_shape)) return native_failure();
        if (!correlated_shape.histories) return fail(UMAMI_ERROR_NOT_IMPLEMENTED,"saved runtime has no directed histories");
    }
    size_t histories=correlated ? correlated_shape.histories : 1;
    size_t value_size=correlated ? sizeof(rusticol_virtual_complex64) : sizeof(double);
    size_t value_alignment=correlated ? _Alignof(rusticol_virtual_complex64) : _Alignof(double);
    size_t h=resolved ? shape.helicities : 1;
    size_t bc=resolved ? shape.born_color_components : 1;
    size_t vc=resolved ? shape.virtual_color_components : 1;
    size_t momentum_width, born_width, virtual_width, required;
    if (!product(shape.external_count,4,&momentum_width) ||
        !product(shape.born_orders,h,&born_width) || !product(born_width,bc,&born_width) || !product(born_width,histories,&born_width) ||
        !product(shape.virtual_orders,h,&virtual_width) || !product(virtual_width,vc,&virtual_width) ||
        !product(virtual_width,shape.epsilon_count,&virtual_width) || !product(virtual_width,histories,&virtual_width))
        return fail(UMAMI_ERROR, "UMAMI layout dimensions overflow size_t");
    if (count) {
        if (!product(stride,momentum_width,&required) ||
            !buffer_valid(in[0],in_capacity[0],required,sizeof(double),_Alignof(double)))
            return fail(UMAMI_ERROR, "invalid or short momentum buffer");
        for (unsigned i=1;i<4;i++) if (in_present & (1u<<i)) {
            size_t width=i==2 ? 1 : stride;
            size_t bytes=i==3 ? sizeof(uint64_t) : sizeof(double);
            size_t alignment=i==3 ? _Alignof(uint64_t) : _Alignof(double);
            if (!buffer_valid(in[i],in_capacity[i],width,bytes,alignment))
                return fail(UMAMI_ERROR, "invalid or short scale/point-ID buffer");
        }
        for (unsigned i=0;i<3;i++) if (out_present & (1u<<i)) {
            size_t width=i==0 ? born_width : i==1 ? virtual_width : 1;
            size_t bytes=i==2 ? sizeof(uint32_t) : value_size;
            size_t alignment=i==2 ? _Alignof(uint32_t) : value_alignment;
            if (!product(stride,width,&required) ||
                !buffer_valid(out[i],out_capacity[i],required,bytes,alignment))
                return fail(UMAMI_ERROR, "invalid or short virtual-extension output buffer");
        }
    }
    size_t momentum_count;
    if (!product(count,momentum_width,&momentum_count))
        return fail(UMAMI_ERROR, "packed momentum layout overflows size_t");
    double *momenta=array(momentum_count,sizeof(double));
    double *scales=array(count,sizeof(double));
    uint64_t *ids=array(count,sizeof(uint64_t)), *success_ids=NULL;
    uint32_t *statuses=array(count,sizeof(uint32_t));
    void *born=NULL, *virtual_values=NULL;
    void *result=NULL;
    UmamiStatus status=UMAMI_SUCCESS;
    if (count && (!momenta || !scales || !ids || !statuses)) {
        status=fail(UMAMI_ERROR,"unable to allocate virtual bulk input buffers"); goto done;
    }
    for (size_t p=0;p<count;p++) {
        size_t event=offset+p;
        for (size_t leg=0;leg<shape.external_count;leg++)
            for (size_t component=0;component<4;component++)
                momenta[p*momentum_width+4*leg+component]=
                    ((const double*)in[0])[event+stride*(leg+shape.external_count*component)];
        scales[p]=in_present & 2u ? ((const double*)in[1])[event] : ((const double*)in[2])[0];
        ids[p]=in_present & 8u ? ((const uint64_t*)in[3])[event] : (uint64_t)event;
    }
    rusticol_virtual_batch_request request={momenta,momentum_count,count,ids,scales,count,resolved,
        options ? options->r2_strategy : NULL, options ? options->verification_json : NULL};
    rusticol_virtual_result_shape_info result_shape={0};
    if (correlated) {
        rusticol_virtual_correlated_result *typed=NULL;
        if (rusticol_virtual_evaluate_correlated(handle,&request,&typed)) { status=native_failure(); goto done; }
        result=typed;
        rusticol_virtual_correlated_result_shape_info axes;
        if (rusticol_virtual_correlated_result_shape(typed,&axes)) { status=native_failure(); goto done; }
        if (axes.axes.histories!=histories || axes.axes.born_orders!=shape.born_orders ||
            axes.axes.virtual_orders!=shape.virtual_orders || axes.axes.epsilon_count!=shape.epsilon_count) {
            status=fail(UMAMI_ERROR,"native correlated catalogue changed during evaluation"); goto done;
        }
        result_shape.requested_points=axes.requested_points; result_shape.successful_points=axes.successful_points;
        result_shape.born_values=axes.born_values; result_shape.virtual_values=axes.virtual_values;
    } else {
        rusticol_virtual_result *typed=NULL;
        if (rusticol_virtual_evaluate(handle,&request,&typed)) { status=native_failure(); goto done; }
        result=typed;
        if (rusticol_virtual_result_shape(typed,&result_shape)) { status=native_failure(); goto done; }
    }
    size_t born_count, virtual_count;
    if (!product(result_shape.successful_points,born_width,&born_count) ||
        !product(result_shape.successful_points,virtual_width,&virtual_count) ||
        result_shape.requested_points!=count || result_shape.successful_points>count ||
        result_shape.born_values!=born_count || result_shape.virtual_values!=virtual_count ||
        result_shape.resolved!=resolved) {
        status=fail(UMAMI_ERROR,"native result does not match the declared UMAMI axes"); goto done;
    }
    success_ids=array(result_shape.successful_points,sizeof(uint64_t));
    if (out_present & 1u) born=array(born_count,value_size);
    if (out_present & 2u) virtual_values=array(virtual_count,value_size);
    if ((result_shape.successful_points && !success_ids) ||
        ((out_present & 1u) && born_count && !born) ||
        ((out_present & 2u) && virtual_count && !virtual_values)) {
        status=fail(UMAMI_ERROR,"unable to allocate virtual bulk result buffers"); goto done;
    }
    int copy_error;
    if (correlated) {
        copy_error=rusticol_virtual_correlated_result_copy_statuses(result,statuses,count) ||
            rusticol_virtual_correlated_result_copy_successful_ids(result,success_ids,result_shape.successful_points) ||
            ((out_present & 1u) && rusticol_virtual_correlated_result_copy_born(result,born,born_count)) ||
            ((out_present & 2u) && rusticol_virtual_correlated_result_copy_virtual(result,virtual_values,virtual_count));
    } else {
        copy_error=rusticol_virtual_result_copy_statuses(result,statuses,count) ||
            rusticol_virtual_result_copy_successful_ids(result,success_ids,result_shape.successful_points) ||
            ((out_present & 1u) && rusticol_virtual_result_copy_born(result,born,born_count)) ||
            ((out_present & 2u) && rusticol_virtual_result_copy_virtual(result,virtual_values,virtual_count));
    }
    if (copy_error) { status=native_failure(); goto done; }
    size_t row=0;
    for (size_t p=0;p<count;p++) {
        if (statuses[p]>RUSTICOL_VIRTUAL_ACCURACY_NOT_ESTABLISHED ||
            (successful(statuses[p]) && (row>=result_shape.successful_points || success_ids[row++]!=ids[p]))) {
            status=fail(UMAMI_ERROR,"native compact rows do not match physical point statuses"); goto done;
        }
    }
    if (row!=result_shape.successful_points) {
        status=fail(UMAMI_ERROR,"native compact row count does not match physical point statuses"); goto done;
    }
    /* Every function-error path is above this point. Only transport writes follow. */
    row=0;
    for (size_t p=0;p<count;p++) {
        size_t event=offset+p;
        ((uint32_t*)out[2])[event]=statuses[p];
        if (!successful(statuses[p])) continue;
        if (correlated) {
            for (size_t history=0;history<histories;history++) {
                if (out_present & 1u) for (size_t o=0;o<shape.born_orders;o++)
                    ((rusticol_virtual_complex64*)out[0])[event+stride*(o+shape.born_orders*history)]=
                        ((rusticol_virtual_complex64*)born)[(row*histories+history)*shape.born_orders+o];
                if (out_present & 2u) for (size_t o=0;o<shape.virtual_orders;o++)
                    for (size_t epsilon=0;epsilon<shape.epsilon_count;epsilon++)
                        ((rusticol_virtual_complex64*)out[1])[event+stride*(o+shape.virtual_orders*(history+histories*epsilon))]=
                            ((rusticol_virtual_complex64*)virtual_values)[((row*histories+history)*shape.virtual_orders+o)*shape.epsilon_count+epsilon];
            }
            row++; continue;
        }
        if (out_present & 1u)
            for (size_t o=0;o<shape.born_orders;o++)
                for (size_t helicity=0;helicity<h;helicity++)
                    for (size_t colour=0;colour<bc;colour++)
                        ((double*)out[0])[event+stride*(o+shape.born_orders*(helicity+h*colour))]=
                            ((double*)born)[((row*shape.born_orders+o)*h+helicity)*bc+colour];
        if (out_present & 2u)
            for (size_t o=0;o<shape.virtual_orders;o++)
                for (size_t helicity=0;helicity<h;helicity++)
                    for (size_t colour=0;colour<vc;colour++)
                        for (size_t epsilon=0;epsilon<shape.epsilon_count;epsilon++)
                            ((double*)out[1])[event+stride*(o+shape.virtual_orders*(helicity+h*(colour+vc*epsilon)))]=
                                ((double*)virtual_values)[(((row*shape.virtual_orders+o)*h+helicity)*vc+colour)*shape.epsilon_count+epsilon];
        row++;
    }
    if (diagnostic_result) { *diagnostic_result=result; result=NULL; }
done:
    if (result) {
        if (correlated) rusticol_virtual_correlated_result_free(result);
        else rusticol_virtual_result_free(result);
    }
    free(momenta); free(scales); free(ids); free(statuses);
    free(success_ids); free(born); free(virtual_values);
    return status;
}

UmamiStatus pyamplicol_umami_virtual_v1_evaluate(
    rusticol_virtual_handle *handle, size_t count, size_t stride, size_t offset,
    size_t input_count, const int *input_keys, const void *const *inputs, const size_t *input_capacities,
    size_t output_count, const int *output_keys, void *const *outputs, const size_t *output_capacities,
    const pyamplicol_umami_virtual_v1_options *options, rusticol_virtual_result **diagnostic_result) {
    void *result=NULL;
    if (diagnostic_result) *diagnostic_result=NULL;
    UmamiStatus status=evaluate_bulk(0,handle,count,stride,offset,input_count,input_keys,inputs,input_capacities,
        output_count,output_keys,outputs,output_capacities,options,diagnostic_result ? &result : NULL);
    if (diagnostic_result) *diagnostic_result=result;
    return status;
}

UmamiStatus pyamplicol_umami_virtual_v1_evaluate_correlated(
    rusticol_virtual_handle *handle, size_t count, size_t stride, size_t offset,
    size_t input_count, const int *input_keys, const void *const *inputs, const size_t *input_capacities,
    size_t output_count, const int *output_keys, void *const *outputs, const size_t *output_capacities,
    const pyamplicol_umami_virtual_v1_options *options, rusticol_virtual_correlated_result **diagnostic_result) {
    void *result=NULL;
    if (diagnostic_result) *diagnostic_result=NULL;
    UmamiStatus status=evaluate_bulk(1,handle,count,stride,offset,input_count,input_keys,inputs,input_capacities,
        output_count,output_keys,outputs,output_capacities,options,diagnostic_result ? &result : NULL);
    if (diagnostic_result) *diagnostic_result=result;
    return status;
}

UmamiStatus pyamplicol_umami_virtual_v1_supported_correlated_outputs(const int **keys, size_t *count) {
    static const int supported[]={PYAMPLICOL_UMAMI_V1_OUT_CORRELATED_BORN,
        PYAMPLICOL_UMAMI_V1_OUT_CORRELATED_VIRTUAL,PYAMPLICOL_UMAMI_V1_OUT_STATUS};
    if (!keys || !count) return fail(UMAMI_ERROR,"null supported-output query output");
    *keys=supported; *count=sizeof(supported)/sizeof(supported[0]); return UMAMI_SUCCESS;
}
