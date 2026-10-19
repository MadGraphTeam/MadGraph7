/* SPDX-License-Identifier: 0BSD */
/* Native ABI control: ./client /path/to/resolved-gg-source-artifact */
#include "rusticol_virtual.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check(int status) {
  if(status != RUSTICOL_VIRTUAL_OK) {
    fprintf(stderr,"%s\n",rusticol_virtual_last_error_message()); exit(1);
  }
}
static void close_value(double a,double b) {
  assert(fabs(a-b) <= 1e-8*fmax(fmax(fabs(a),fabs(b)),1e-20));
}
int main(int argc,char **argv) {
  if(argc != 2) { fprintf(stderr,"usage: %s artifact\n",argv[0]); return 2; }
  assert(rusticol_virtual_abi_version()==1);
  rusticol_virtual_handle *runtime=NULL;
  check(rusticol_virtual_open(argv[1],"gg",NULL,"model",&runtime));
  rusticol_virtual_shape_info shape;
  check(rusticol_virtual_shape(runtime,&shape));
  assert(shape.external_count==4 && shape.epsilon_count==3 && shape.resolved_available);
  size_t size=0;
  check(rusticol_virtual_metadata_json(runtime,NULL,0,&size));
  char *metadata=malloc(size); assert(metadata);
  check(rusticol_virtual_metadata_json(runtime,metadata,size,&size));
  assert(strstr(metadata,"\"epsilon_powers\":[0,-1,-2]")); free(metadata);
  const double p[16]={500,0,0,500,500,0,0,-500,500,300,0,400,500,-300,0,-400};
  double momenta[48]; for(int i=0;i<3;i++) memcpy(momenta+i*16,p,sizeof(p));
  uint64_t ids[3]={91,4,70}; double scales[3]={1e6,9e5,1e6};
  rusticol_virtual_batch_request request={momenta,48,3,ids,scales,3,0,NULL,NULL};
  rusticol_virtual_result *inclusive=NULL,*resolved=NULL;
  check(rusticol_virtual_evaluate(runtime,&request,&inclusive));
  rusticol_virtual_result_shape_info info;
  check(rusticol_virtual_result_shape(inclusive,&info));
  assert(info.requested_points==3 && info.successful_points==3 && info.born_values==3 && info.virtual_values==9);
  uint64_t actual_ids[3]; uint32_t statuses[3];
  check(rusticol_virtual_result_copy_successful_ids(inclusive,actual_ids,3));
  check(rusticol_virtual_result_copy_statuses(inclusive,statuses,3));
  assert(!memcmp(ids,actual_ids,sizeof(ids)));
  for(int i=0;i<3;i++) assert(statuses[i]==RUSTICOL_VIRTUAL_SUCCESS);
  double born[3],values[9];
  check(rusticol_virtual_result_copy_born(inclusive,born,3));
  check(rusticol_virtual_result_copy_virtual(inclusive,values,9));
  double canary[3]={71,72,73};
  assert(rusticol_virtual_result_copy_virtual(inclusive,canary,2)==RUSTICOL_VIRTUAL_BUFFER_TOO_SMALL);
  assert(canary[0]==71 && canary[1]==72 && canary[2]==73);
  const double pi=acos(-1.);
  close_value(values[2],-12.*0.118/(2*pi)*born[0]);
  request.resolved=1;
  check(rusticol_virtual_evaluate(runtime,&request,&resolved));
  check(rusticol_virtual_result_shape(resolved,&info));
  double *rb=malloc(info.born_values*sizeof(double)),*rv=malloc(info.virtual_values*sizeof(double)); assert(rb && rv);
  check(rusticol_virtual_result_copy_born(resolved,rb,info.born_values));
  check(rusticol_virtual_result_copy_virtual(resolved,rv,info.virtual_values));
  for(size_t point=0;point<3;point++) {
    double b=0,v[3]={0,0,0};
    for(size_t h=0;h<shape.helicities;h++) {
      for(size_t c=0;c<shape.born_color_components;c++)
        b+=rb[(point*shape.helicities+h)*shape.born_color_components+c];
      for(size_t c=0;c<shape.virtual_color_components;c++)
        for(size_t pole=0;pole<3;pole++)
          v[pole]+=rv[((point*shape.helicities+h)*shape.virtual_color_components+c)*3+pole];
    }
    close_value(b,born[point]); for(size_t pole=0;pole<3;pole++) close_value(v[pole],values[point*3+pole]);
  }
  free(rb); free(rv);
  /* Finite, exactly supplied large inputs trigger a real lower-carrier
   * numerical failure; the successful physical rows stay compact. */
  for(size_t i=0;i<16;i++) momenta[16+i]=p[i]*1e200;
  rusticol_virtual_result *partial=NULL;
  check(rusticol_virtual_evaluate(runtime,&request,&partial));
  check(rusticol_virtual_result_shape(partial,&info));
  assert(info.requested_points==3 && info.successful_points==2);
  check(rusticol_virtual_result_copy_successful_ids(partial,actual_ids,3));
  check(rusticol_virtual_result_copy_statuses(partial,statuses,3));
  assert(actual_ids[0]==91 && actual_ids[1]==70);
  assert(statuses[0]==RUSTICOL_VIRTUAL_SUCCESS && statuses[1]==RUSTICOL_VIRTUAL_FAILED && statuses[2]==RUSTICOL_VIRTUAL_SUCCESS);
  check(rusticol_virtual_result_free(partial));
  memcpy(momenta+16,p,sizeof(p));
  request.point_count=1; request.momentum_count=16; request.scale_count=1;
  request.verification_json="{\"higher\":[{\"arithmetic\":\"double-double\",\"precision\":null}],\"relative_tolerance\":\"1e-8\",\"absolute_tolerance\":\"1e-9\",\"verify_successful\":true}";
  rusticol_virtual_result *verified=NULL;
  check(rusticol_virtual_evaluate(runtime,&request,&verified));
  check(rusticol_virtual_result_copy_statuses(verified,statuses,3));
  assert(statuses[0]==RUSTICOL_VIRTUAL_VERIFIED);
  check(rusticol_virtual_result_free(verified));
  request.point_count=0; request.momentum_count=0; request.scale_count=0;
  request.momenta=NULL; request.mu_r2=NULL; request.point_ids=NULL;
  request.verification_json=NULL;
  rusticol_virtual_result *empty=NULL;
  check(rusticol_virtual_evaluate(runtime,&request,&empty));
  check(rusticol_virtual_result_shape(empty,&info));
  assert(info.requested_points==0 && info.successful_points==0 && info.born_values==0 && info.virtual_values==0);
  check(rusticol_virtual_result_copy_born(empty,NULL,0));
  check(rusticol_virtual_result_free(empty));
  uint64_t revision;
  check(rusticol_virtual_update_parameters(runtime,"{\"aS\":[\"0.12\",\"0\"]}",&revision));
  assert(revision>0);
  check(rusticol_virtual_clear(runtime));
  check(rusticol_virtual_free(runtime));
  /* Result ownership is independent of the now-destroyed runtime. */
  check(rusticol_virtual_result_copy_born(inclusive,born,3));
  check(rusticol_virtual_result_free(inclusive));
  check(rusticol_virtual_result_free(resolved));
  puts("native C virtual order/pole/resolved/buffer/lifetime controls passed");
  return 0;
}
