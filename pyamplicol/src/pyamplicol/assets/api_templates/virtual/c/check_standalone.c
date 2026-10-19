/* SPDX-License-Identifier: 0BSD */
#include "rusticol_virtual.h"
#include "../common/input.h"
static void check(int status) { if(status) pa_die(rusticol_virtual_last_error_message()); }
int main(int argc,char **argv) {
  double scale; int resolved=pa_args(argc,argv,&scale);
  rusticol_virtual_handle *runtime=NULL;
  check(rusticol_virtual_open(".",argv[1],NULL,NULL,&runtime));
  rusticol_virtual_shape_info axes; check(rusticol_virtual_shape(runtime,&axes));
  pa_points points=pa_read_points(argv[1],axes.external_count);
  size_t needed=0;
  check(rusticol_virtual_metadata_json(runtime,NULL,0,&needed));
  char *metadata=(char *)pa_array(needed,1);
  check(rusticol_virtual_metadata_json(runtime,metadata,needed,&needed));
  puts(metadata); free(metadata);
  rusticol_virtual_batch_request request={points.momenta,pa_product(pa_product(points.count,axes.external_count),4),points.count,NULL,&scale,1,(uint32_t)resolved,NULL,NULL};
  rusticol_virtual_result *result=NULL;
  check(rusticol_virtual_evaluate(runtime,&request,&result)); free(points.momenta);
  rusticol_virtual_result_shape_info shape; check(rusticol_virtual_result_shape(result,&shape));
  double *born=(double *)pa_array(shape.born_values,sizeof(double));
  double *loop=(double *)pa_array(shape.virtual_values,sizeof(double));
  uint64_t *ids=(uint64_t *)pa_array(shape.successful_points,sizeof(uint64_t));
  uint32_t *statuses=(uint32_t *)pa_array(shape.requested_points,sizeof(uint32_t));
  check(rusticol_virtual_result_copy_born(result,born,shape.born_values));
  check(rusticol_virtual_result_copy_virtual(result,loop,shape.virtual_values));
  check(rusticol_virtual_result_copy_successful_ids(result,ids,shape.successful_points));
  check(rusticol_virtual_result_copy_statuses(result,statuses,shape.requested_points));
  printf("shape requested=%zu successful=%zu born_orders=%zu virtual_orders=%zu helicities=%zu born_colours=%zu virtual_colours=%zu epsilon=3 resolved=%u\n",shape.requested_points,shape.successful_points,axes.born_orders,axes.virtual_orders,resolved?axes.helicities:1,resolved?axes.born_color_components:1,resolved?axes.virtual_color_components:1,shape.resolved);
  pa_ids("successful_ids",ids,shape.successful_points);
  printf("statuses"); for(size_t i=0;i<shape.requested_points;i++) printf(" %u",statuses[i]); puts("");
  pa_values("born",born,shape.born_values); pa_values("virtual",loop,shape.virtual_values);
  check(rusticol_virtual_result_diagnostics_json(result,NULL,0,&needed));
  char *diagnostics=(char *)pa_array(needed,1);
  check(rusticol_virtual_result_diagnostics_json(result,diagnostics,needed,&needed)); puts(diagnostics);
  free(diagnostics); free(born); free(loop); free(ids); free(statuses);
  check(rusticol_virtual_result_free(result)); check(rusticol_virtual_free(runtime));
  return 0;
}
