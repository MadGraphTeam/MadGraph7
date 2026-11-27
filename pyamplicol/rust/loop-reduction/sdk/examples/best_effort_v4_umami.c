/* SPDX-License-Identifier: 0BSD */
/* The UMAMI provider with a best-effort point: ./best_effort_v4_umami
 * artifact process input.txt [target]. Input: "points legs", then per point its
 * 4*legs momenta and mu_R^2 (C99 hexadecimal). The handle evaluates on NinjaRS
 * with poles "none", the lane rescue off and the reduction target `target`
 * when given (a lane NinjaRS's cancellation rescue refuses then leaves its
 * point to the failure policy, best-effort by default). Output: "status s0 s1 ..." and per point "finite <event>
 * <hex values of each virtual order>" for every point with a value. */
#include "pyamplicol_umami_virtual_v1.h"
#include <stdio.h>
#include <stdlib.h>

static void native_check(int status) {
    if (status) { printf("error SDK: %s\n",rusticol_virtual_last_error_message()); exit(1); }
}
int main(int argc,char **argv) {
    if(argc!=4 && argc!=5) return 2;
    FILE *in=fopen(argv[3],"r");
    if(!in) return 2;
    size_t points=0,legs=0;
    if(fscanf(in,"%zu %zu",&points,&legs)!=2) return 2;
    double *momenta=malloc(points*legs*4*sizeof(double)), *scales=malloc(points*sizeof(double));
    char text[128];
    for(size_t p=0;p<points;p++) {
        for(size_t i=0;i<legs*4;i++) {
            if(fscanf(in,"%127s",text)!=1) return 2;
            /* UMAMI layout: [event + stride*(leg + legs*component)] */
            momenta[p+points*((i/4)+legs*(i%4))]=strtod(text,NULL);
        }
        if(fscanf(in,"%127s",text)!=1) return 2;
        scales[p]=strtod(text,NULL);
    }
    fclose(in);
    rusticol_virtual_handle *runtime=NULL;
    native_check(rusticol_virtual_open(argv[1],argv[2],NULL,NULL,&runtime));
    native_check(rusticol_virtual_set_poles(runtime,"none"));
    native_check(rusticol_virtual_set_reduction(runtime,"ninjars"));
    native_check(rusticol_virtual_set_lane_rescue(runtime,0));
    if(argc==5) native_check(rusticol_virtual_set_reduction_target(runtime,strtod(argv[4],NULL)));
    rusticol_virtual_shape_info shape;
    native_check(rusticol_virtual_shape(runtime,&shape));
    const size_t width=shape.virtual_orders*3;
    double *virt=malloc(points*width*sizeof(double));
    uint32_t *statuses=malloc(points*sizeof(uint32_t));
    for(size_t i=0;i<points*width;i++) virt[i]=-1234567.0;
    const int input_keys[2]={UMAMI_IN_MOMENTA,PYAMPLICOL_UMAMI_V1_IN_MU_R2};
    const void *inputs[2]={momenta,scales};
    const size_t input_capacities[2]={points*legs*4,points};
    const int output_keys[2]={PYAMPLICOL_UMAMI_V1_OUT_VIRTUAL,PYAMPLICOL_UMAMI_V1_OUT_STATUS};
    void *outputs[2]={virt,statuses};
    const size_t output_capacities[2]={points*width,points};
    pyamplicol_umami_virtual_v1_options options={0,NULL};
    rusticol_virtual_result *details=NULL;
    UmamiStatus status=pyamplicol_umami_virtual_v1_evaluate(runtime,points,points,0,2,input_keys,inputs,
        input_capacities,2,output_keys,outputs,output_capacities,&options,&details);
    if(status!=UMAMI_SUCCESS) {
        printf("error extension: %s\n",pyamplicol_umami_virtual_v1_last_error_message()); return 1;
    }
    printf("status");
    for(size_t p=0;p<points;p++) printf(" %u",statuses[p]);
    printf("\n");
    for(size_t p=0;p<points;p++) {
        if(statuses[p]==RUSTICOL_VIRTUAL_FAILED) continue;
        printf("finite %zu",p);
        for(size_t o=0;o<shape.virtual_orders;o++) printf(" %a",virt[p+points*o]);
        printf("\n");
    }
    rusticol_virtual_result_free(details);
    rusticol_virtual_free(runtime);
    free(momenta); free(scales); free(virt); free(statuses);
    return 0;
}
