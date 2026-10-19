/* SPDX-License-Identifier: 0BSD */
/* Native extension control: ./client-umami ARTIFACT [uudd|gg]. */
#include "pyamplicol_umami_virtual_v1.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void native_check(int status) {
    if (status) { fprintf(stderr,"SDK: %s\n",rusticol_virtual_last_error_message()); exit(1); }
}
static void check(UmamiStatus status) {
    if (status!=UMAMI_SUCCESS) {
        fprintf(stderr,"extension: %s\n",pyamplicol_umami_virtual_v1_last_error_message()); exit(1);
    }
}
static void close_value(double a,double b) {
    if (fabs(a-b)>1e-8*fmax(fmax(fabs(a),fabs(b)),1e-20)) {
        fprintf(stderr,"mismatch %.17g versus %.17g\n",a,b); exit(1);
    }
}
static const double point[16]={500,0,0,500,500,0,0,-500,500,300,0,400,500,-300,0,-400};
static const double sentinel=-1234567.0;
static void fill(double *values,size_t count) { for(size_t i=0;i<count;i++) values[i]=sentinel; }
int main(int argc,char **argv) {
    if(argc<2 || argc>3) return 2;
    const char *process=argc==3 ? argv[2] : "uudd";
    rusticol_virtual_handle *runtime=NULL;
    native_check(rusticol_virtual_open(argv[1],process,NULL,"model",&runtime));
    rusticol_virtual_shape_info shape;
    native_check(rusticol_virtual_shape(runtime,&shape));
    assert(shape.external_count==4 && shape.resolved_available && shape.epsilon_count==3);
    if(shape.virtual_color_components>1) {
        size_t bytes=0;
        native_check(rusticol_virtual_capabilities_json(runtime,NULL,0,&bytes));
        char *capabilities=malloc(bytes); assert(capabilities);
        native_check(rusticol_virtual_capabilities_json(runtime,capabilities,bytes,&bytes));
        assert(strstr(capabilities,"signed canonical Born tensor groups"));
        free(capabilities);
    }
    assert(pyamplicol_umami_virtual_v1_version()==1);
    const int *supported=NULL; size_t nkeys=0;
    check(pyamplicol_umami_virtual_v1_supported_inputs(&supported,&nkeys)); assert(nkeys==4 && supported[0]==UMAMI_IN_MOMENTA);
    check(pyamplicol_umami_virtual_v1_supported_outputs(&supported,&nkeys)); assert(nkeys==3);
    for(size_t k=0;k<nkeys;k++) assert(supported[k]!=UMAMI_OUT_MATRIX_ELEMENT);
    const size_t stride=7, offset=2, count=3;
    double momenta[7*16], scales[7], scalar_scale=1e6;
    uint64_t ids[7]={100,101,91,4,70,105,106};
    uint32_t statuses[7];
    for(size_t i=0;i<7*16;i++) momenta[i]=NAN; /* outside-slice values must not be read */
    for(size_t i=0;i<7;i++) scales[i]=NAN;
    for(size_t event=offset;event<offset+count;event++) {
        scales[event]=event==3 ? 9e5 : 1e6;
        for(size_t leg=0;leg<4;leg++) for(size_t component=0;component<4;component++)
            momenta[event+stride*(leg+4*component)]=point[4*leg+component];
    }
    const int input_keys[3]={UMAMI_IN_MOMENTA,PYAMPLICOL_UMAMI_V1_IN_MU_R2,PYAMPLICOL_UMAMI_V1_IN_POINT_ID};
    const void *inputs[3]={momenta,scales,ids};
    const size_t input_capacities[3]={7*16,7,7};
    const int output_keys[3]={PYAMPLICOL_UMAMI_V1_OUT_BORN,PYAMPLICOL_UMAMI_V1_OUT_VIRTUAL,PYAMPLICOL_UMAMI_V1_OUT_STATUS};
    double *inclusive_born=NULL,*inclusive_virtual=NULL;
    for(unsigned resolved=0;resolved<2;resolved++) {
        size_t h=resolved ? shape.helicities : 1;
        size_t bc=resolved ? shape.born_color_components : 1;
        size_t vc=resolved ? shape.virtual_color_components : 1;
        size_t bw=shape.born_orders*h*bc, vw=shape.virtual_orders*h*vc*3;
        double *born=malloc(stride*bw*sizeof(double)), *virt=malloc(stride*vw*sizeof(double));
        assert(born && virt); fill(born,stride*bw); fill(virt,stride*vw);
        for(size_t i=0;i<stride;i++) statuses[i]=123;
        void *outputs[3]={born,virt,statuses};
        size_t output_capacities[3]={stride*bw,stride*vw,stride};
        pyamplicol_umami_virtual_v1_options options={resolved,NULL,NULL};
        rusticol_virtual_result *details=NULL;
        check(pyamplicol_umami_virtual_v1_evaluate(runtime,count,stride,offset,3,input_keys,inputs,input_capacities,
            3,output_keys,outputs,output_capacities,&options,&details));
        assert(details);
        uint64_t returned_ids[3];
        native_check(rusticol_virtual_result_copy_requested_ids(details,returned_ids,3));
        assert(returned_ids[0]==91 && returned_ids[1]==4 && returned_ids[2]==70);
        double *reference_born=malloc(count*bw*sizeof(double)), *reference_virtual=malloc(count*vw*sizeof(double));
        assert(reference_born && reference_virtual);
        native_check(rusticol_virtual_result_copy_born(details,reference_born,count*bw));
        native_check(rusticol_virtual_result_copy_virtual(details,reference_virtual,count*vw));
        for(size_t event=0;event<stride;event++) {
            if(event<offset || event>=offset+count) {
                assert(statuses[event]==123);
                for(size_t k=0;k<bw;k++) assert(born[event+stride*k]==sentinel);
                for(size_t k=0;k<vw;k++) assert(virt[event+stride*k]==sentinel);
                continue;
            }
            assert(statuses[event]==RUSTICOL_VIRTUAL_SUCCESS);
            size_t p=event-offset;
            for(size_t o=0;o<shape.born_orders;o++) for(size_t hel=0;hel<h;hel++) for(size_t c=0;c<bc;c++)
                close_value(born[event+stride*(o+shape.born_orders*(hel+h*c))],reference_born[((p*shape.born_orders+o)*h+hel)*bc+c]);
            for(size_t o=0;o<shape.virtual_orders;o++) for(size_t hel=0;hel<h;hel++) for(size_t c=0;c<vc;c++) for(size_t e=0;e<3;e++)
                close_value(virt[event+stride*(o+shape.virtual_orders*(hel+h*(c+vc*e)))],reference_virtual[(((p*shape.virtual_orders+o)*h+hel)*vc+c)*3+e]);
        }
        if(!resolved) {
            inclusive_born=malloc(count*shape.born_orders*sizeof(double));
            inclusive_virtual=malloc(count*shape.virtual_orders*3*sizeof(double));
            assert(inclusive_born && inclusive_virtual);
            memcpy(inclusive_born,reference_born,count*bw*sizeof(double));
            memcpy(inclusive_virtual,reference_virtual,count*vw*sizeof(double));
            if(!strcmp(process,"uudd")) {
                assert(shape.born_orders==3 && shape.virtual_orders==3);
                /* Independent MadLoop fixture, actual independent catalogue ordering. */
                const double expected_born[3]={-2.0322878122576093,0.8013367806451908,3.0512301482522957};
                const double expected[9]={-3.018755956775497,-1.1476469718544264,-0.3056158216076459,
                    1.313041990604555,0.7811563156803705,0.20355701776284274,
                    -0.19094989612756078,-0.28421984016919477,-0.08026310265110026};
                for(size_t o=0;o<3;o++) close_value(reference_born[o],expected_born[o]);
                for(size_t k=0;k<9;k++) close_value(reference_virtual[k],expected[k]);
            }
        } else {
            for(size_t p=0;p<count;p++) {
                for(size_t o=0;o<shape.born_orders;o++) {
                    double sum=0; for(size_t hel=0;hel<h;hel++) for(size_t c=0;c<bc;c++) sum+=reference_born[((p*shape.born_orders+o)*h+hel)*bc+c];
                    close_value(sum,inclusive_born[p*shape.born_orders+o]);
                }
                for(size_t o=0;o<shape.virtual_orders;o++) for(size_t e=0;e<3;e++) {
                    double sum=0; for(size_t hel=0;hel<h;hel++) for(size_t c=0;c<vc;c++) sum+=reference_virtual[(((p*shape.virtual_orders+o)*h+hel)*vc+c)*3+e];
                    close_value(sum,inclusive_virtual[(p*shape.virtual_orders+o)*3+e]);
                }
            }
        }
        native_check(rusticol_virtual_result_free(details)); details=NULL;
        free(reference_born); free(reference_virtual);
        /* Output layout failure is atomic, even when earlier outputs are valid. */
        fill(born,stride*bw); fill(virt,stride*vw); for(size_t i=0;i<stride;i++) statuses[i]=123;
        output_capacities[1]--;
        assert(pyamplicol_umami_virtual_v1_evaluate(runtime,count,stride,offset,3,input_keys,inputs,input_capacities,
            3,output_keys,outputs,output_capacities,&options,&details)==UMAMI_ERROR && !details);
        for(size_t i=0;i<stride*bw;i++) assert(born[i]==sentinel);
        for(size_t i=0;i<stride*vw;i++) assert(virt[i]==sentinel);
        for(size_t i=0;i<stride;i++) assert(statuses[i]==123);
        output_capacities[1]++;
        /* Real numerical failed middle point, rather than an injected transport failure. */
        for(size_t leg=0;leg<4;leg++) for(size_t component=0;component<4;component++)
            momenta[3+stride*(leg+4*component)]=point[4*leg+component]*1e200;
        check(pyamplicol_umami_virtual_v1_evaluate(runtime,count,stride,offset,3,input_keys,inputs,input_capacities,
            3,output_keys,outputs,output_capacities,&options,&details));
        assert(statuses[2]==0 && statuses[3]==3 && statuses[4]==0);
        for(size_t k=0;k<bw;k++) assert(born[3+stride*k]==sentinel);
        for(size_t k=0;k<vw;k++) assert(virt[3+stride*k]==sentinel);
        rusticol_virtual_result_shape_info rs;
        native_check(rusticol_virtual_result_shape(details,&rs)); assert(rs.successful_points==2);
        native_check(rusticol_virtual_result_copy_successful_ids(details,returned_ids,3));
        assert(returned_ids[0]==91 && returned_ids[1]==70);
        native_check(rusticol_virtual_result_free(details)); details=NULL;
        check(pyamplicol_umami_virtual_v1_evaluate(runtime,1,stride,3,3,input_keys,inputs,input_capacities,
            3,output_keys,outputs,output_capacities,&options,&details));
        native_check(rusticol_virtual_result_shape(details,&rs));
        assert(rs.requested_points==1 && rs.successful_points==0 && rs.born_values==0 && rs.virtual_values==0);
        for(size_t k=0;k<bw;k++) assert(born[3+stride*k]==sentinel);
        for(size_t k=0;k<vw;k++) assert(virt[3+stride*k]==sentinel);
        native_check(rusticol_virtual_result_free(details)); details=NULL;
        for(size_t leg=0;leg<4;leg++) for(size_t component=0;component<4;component++)
            momenta[3+stride*(leg+4*component)]=point[4*leg+component];
        /* Scalar scale, default original event IDs, explicit native verification. */
        const int scalar_keys[2]={UMAMI_IN_MOMENTA,PYAMPLICOL_UMAMI_V1_IN_MU_R2_SCALAR};
        const void *scalar_inputs[2]={momenta,&scalar_scale}; size_t scalar_capacities[2]={7*16,1};
        options.verification_json="{\"higher\":[{\"arithmetic\":\"double-double\",\"precision\":null}],\"relative_tolerance\":\"1e-8\",\"absolute_tolerance\":\"1e-9\",\"verify_successful\":true}";
        check(pyamplicol_umami_virtual_v1_evaluate(runtime,1,stride,offset,2,scalar_keys,scalar_inputs,scalar_capacities,
            3,output_keys,outputs,output_capacities,&options,&details));
        assert(statuses[offset]==RUSTICOL_VIRTUAL_VERIFIED);
        native_check(rusticol_virtual_result_copy_requested_ids(details,returned_ids,3)); assert(returned_ids[0]==offset);
        native_check(rusticol_virtual_result_free(details)); details=NULL;
        options.verification_json=NULL;
        /* Missing mandatory statuses, unsupported scalar tree output, duplicates. */
        assert(pyamplicol_umami_virtual_v1_evaluate(runtime,count,stride,offset,3,input_keys,inputs,input_capacities,
            2,output_keys,outputs,output_capacities,&options,NULL)==UMAMI_ERROR);
        int bad_key=UMAMI_OUT_MATRIX_ELEMENT;
        assert(pyamplicol_umami_virtual_v1_evaluate(runtime,0,0,0,0,NULL,NULL,NULL,
            1,&bad_key,outputs,output_capacities,&options,NULL)==UMAMI_ERROR_UNSUPPORTED_OUTPUT);
        int duplicates[2]={UMAMI_IN_MOMENTA,UMAMI_IN_MOMENTA};
        assert(pyamplicol_umami_virtual_v1_evaluate(runtime,count,stride,offset,2,duplicates,inputs,input_capacities,
            3,output_keys,outputs,output_capacities,&options,NULL)==UMAMI_ERROR);
        assert(pyamplicol_umami_virtual_v1_evaluate(runtime,2,1,0,0,NULL,NULL,NULL,
            0,NULL,NULL,NULL,&options,NULL)==UMAMI_ERROR);
        size_t overflow_capacities[3]={SIZE_MAX,SIZE_MAX,SIZE_MAX};
        assert(pyamplicol_umami_virtual_v1_evaluate(runtime,1,SIZE_MAX,0,3,input_keys,inputs,overflow_capacities,
            3,output_keys,outputs,overflow_capacities,&options,NULL)==UMAMI_ERROR);
        /* Empty calls still use the existing native empty result and ownership. */
        check(pyamplicol_umami_virtual_v1_evaluate(runtime,0,0,0,0,NULL,NULL,NULL,
            0,NULL,NULL,NULL,&options,&details));
        native_check(rusticol_virtual_result_shape(details,&rs)); assert(rs.requested_points==0 && rs.born_values==0);
        native_check(rusticol_virtual_result_free(details)); details=NULL;
        options.r2_strategy="direct";
        assert(pyamplicol_umami_virtual_v1_evaluate(runtime,0,0,0,0,NULL,NULL,NULL,
            0,NULL,NULL,NULL,&options,&details)==UMAMI_ERROR && !details);
        free(born); free(virt);
    }
    uint64_t revision;
    native_check(rusticol_virtual_update_parameters(runtime,"{\"aS\":[\"0.12\",\"0\"]}",&revision)); assert(revision>0);
    native_check(rusticol_virtual_clear(runtime)); native_check(rusticol_virtual_free(runtime));
    free(inclusive_born); free(inclusive_virtual);
    printf("native UMAMI virtual v1 %s order/pole/resolved/stride/failure/verification controls passed\n",process);
    return 0;
}
