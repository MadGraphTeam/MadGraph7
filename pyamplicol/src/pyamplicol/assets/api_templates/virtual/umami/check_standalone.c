/* SPDX-License-Identifier: 0BSD */
/* Versioned pyAmpliCol extension; this exports no standard UMAMI virtual ABI. */
#include "pyamplicol_umami_virtual_v1.h"
#include "../common/input.h"
static int pa_umami_evaluate(rusticol_virtual_handle *runtime,
    const rusticol_virtual_batch_request *request,rusticol_virtual_result **result) {
  rusticol_virtual_shape_info axes;
  int status=rusticol_virtual_shape(runtime,&axes); if(status) return status;
  size_t count=request->point_count;
  if(count>SIZE_MAX-2) pa_die("UMAMI stride overflow");
  const size_t offset=1,stride=count+2;
  size_t inputs=pa_product(pa_product(stride,axes.external_count),4);
  size_t hc=request->resolved?axes.helicities:1;
  size_t born_columns=pa_product(axes.born_orders,pa_product(hc,request->resolved?axes.born_color_components:1));
  size_t virtual_columns=pa_product(pa_product(axes.virtual_orders,3),pa_product(hc,request->resolved?axes.virtual_color_components:1));
  size_t nb=pa_product(stride,born_columns),nv=pa_product(stride,virtual_columns);
  double *momenta=(double *)pa_array(inputs,sizeof(double));
  uint64_t *ids=(uint64_t *)pa_array(stride,sizeof(uint64_t));
  double *born=(double *)pa_array(nb,sizeof(double)),*loop=(double *)pa_array(nv,sizeof(double));
  uint32_t *statuses=(uint32_t *)pa_array(stride,sizeof(uint32_t));
  for(size_t p=0;p<count;p++) {
    ids[offset+p]=request->point_ids?request->point_ids[p]:(uint64_t)p;
    for(size_t leg=0;leg<axes.external_count;leg++)
      for(size_t component=0;component<4;component++)
        momenta[offset+p+stride*(leg+axes.external_count*component)]=request->momenta[(p*axes.external_count+leg)*4+component];
  }
  const int in_keys[]={UMAMI_IN_MOMENTA,PYAMPLICOL_UMAMI_V1_IN_MU_R2_SCALAR,PYAMPLICOL_UMAMI_V1_IN_POINT_ID};
  const void *in[]={momenta,request->mu_r2,ids}; const size_t in_sizes[]={inputs,1,stride};
  const int out_keys[]={PYAMPLICOL_UMAMI_V1_OUT_BORN,PYAMPLICOL_UMAMI_V1_OUT_VIRTUAL,PYAMPLICOL_UMAMI_V1_OUT_STATUS};
  void *out[]={born,loop,statuses}; const size_t out_sizes[]={nb,nv,stride};
  pyamplicol_umami_virtual_v1_options options={request->resolved,request->r2_strategy,request->verification_json};
  UmamiStatus call=pyamplicol_umami_virtual_v1_evaluate(runtime,count,stride,offset,3,in_keys,in,in_sizes,3,out_keys,out,out_sizes,&options,result);
  if(call!=UMAMI_SUCCESS) pa_die(pyamplicol_umami_virtual_v1_last_error_message());
  puts("umami per-point axes Born=[colour,helicity,order], virtual=[epsilon,colour,helicity,order]; order varies fastest");
  /* These are the actual UMAMI column-major outputs. Failed slots are never
   * read: statuses cover every input, while numerical outputs are partial. */
  for(size_t p=0;p<count;p++) {
    printf("umami status point=%" PRIu64 " %u\n",ids[offset+p],statuses[offset+p]);
    if(statuses[offset+p]>RUSTICOL_VIRTUAL_RESCUED) continue;
    printf("umami born point=%" PRIu64,ids[offset+p]);
    for(size_t column=0;column<born_columns;column++) printf(" %.17g",born[offset+p+stride*column]);
    puts("");
    printf("umami virtual point=%" PRIu64,ids[offset+p]);
    for(size_t column=0;column<virtual_columns;column++) printf(" %.17g",loop[offset+p+stride*column]);
    puts("");
  }
  free(momenta); free(ids); free(born); free(loop); free(statuses);
  return 0;
}
/* Reuse the ordinary SDK metadata/owned diagnostic printer around the one
 * UMAMI bulk call above; there is no second numerical evaluation. */
#define rusticol_virtual_evaluate pa_umami_evaluate
#include "../c/check_standalone.c"
