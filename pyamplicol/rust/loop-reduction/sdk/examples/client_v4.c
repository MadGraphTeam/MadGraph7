/* SPDX-License-Identifier: 0BSD */
/* Plan v4 control (WP3.1 I6): ./client_v4 artifact process points.txt
 *
 * points.txt: "points legs", then per point 4*legs momentum components and
 * its mu_R^2, as C99 hexadecimal or decimal floating literals. Prints, per
 * point, its status, then the Born per Born order and the complete virtual
 * per sector and epsilon power (finite, 1/eps, 1/eps^2) as hexadecimal
 * literals ("%a"), so a caller compares them bitwise with the Python API
 * (tools/loop_v2_sdk_check.py). */
#include "rusticol_virtual.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check(int status) {
  if(status != RUSTICOL_VIRTUAL_OK) {
    fprintf(stderr,"%s\n",rusticol_virtual_last_error_message()); exit(1);
  }
}
static double next_value(FILE *input) {
  char text[128];
  if(fscanf(input,"%127s",text) != 1) { fprintf(stderr,"short input\n"); exit(2); }
  return strtod(text,NULL);
}
int main(int argc,char **argv) {
  if(argc != 4) { fprintf(stderr,"usage: %s artifact process points\n",argv[0]); return 2; }
  FILE *input=fopen(argv[3],"r");
  if(!input) { perror(argv[3]); return 2; }
  size_t points=0,legs=0;
  if(fscanf(input,"%zu %zu",&points,&legs) != 2 || !points || !legs) return 2;
  double *momenta=malloc(points*legs*4*sizeof(double)),*scales=malloc(points*sizeof(double));
  if(!momenta || !scales) return 2;
  for(size_t p=0;p<points;p++) {
    for(size_t i=0;i<legs*4;i++) momenta[p*legs*4+i]=next_value(input);
    scales[p]=next_value(input);
  }
  fclose(input);
  rusticol_virtual_handle *runtime=NULL;
  check(rusticol_virtual_open(argv[1],argv[2],NULL,"model",&runtime));
  rusticol_virtual_shape_info shape;
  check(rusticol_virtual_shape(runtime,&shape));
  if(shape.external_count != legs || shape.epsilon_count != 3) { fprintf(stderr,"shape\n"); return 1; }
  rusticol_virtual_batch_request request={momenta,points*legs*4,points,NULL,scales,points,0,NULL,NULL};
  rusticol_virtual_result *result=NULL;
  check(rusticol_virtual_evaluate(runtime,&request,&result));
  rusticol_virtual_result_shape_info info;
  check(rusticol_virtual_result_shape(result,&info));
  uint32_t *statuses=malloc(points*sizeof(uint32_t));
  double *born=malloc((info.born_values+1)*sizeof(double)),*values=malloc((info.virtual_values+1)*sizeof(double));
  if(!statuses || !born || !values) return 2;
  check(rusticol_virtual_result_copy_statuses(result,statuses,points));
  check(rusticol_virtual_result_copy_born(result,born,info.born_values));
  check(rusticol_virtual_result_copy_virtual(result,values,info.virtual_values));
  size_t row=0;
  for(size_t p=0;p<points;p++) {
    printf("point %zu status %u\n",p,statuses[p]);
    if(statuses[p] == RUSTICOL_VIRTUAL_FAILED) continue;
    printf("born");
    for(size_t o=0;o<shape.born_orders;o++) printf(" %a",born[row*shape.born_orders+o]);
    printf("\nvirtual");
    for(size_t s=0;s<shape.virtual_orders*3;s++) printf(" %a",values[row*shape.virtual_orders*3+s]);
    printf("\n");
    row++;
  }
  check(rusticol_virtual_result_free(result));
  check(rusticol_virtual_free(runtime));
  free(momenta); free(scales); free(statuses); free(born); free(values);
  return 0;
}
