/* SPDX-License-Identifier: 0BSD */
/* Validation-point ingress shared by the C, C++ and UMAMI examples. */
#ifndef PYAMPLICOL_VIRTUAL_EXAMPLE_INPUT_H
#define PYAMPLICOL_VIRTUAL_EXAMPLE_INPUT_H
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void pa_die(const char *message) { fprintf(stderr,"%s\n",message); exit(1); }
static size_t pa_product(size_t a,size_t b) {
  if(b && a>SIZE_MAX/b) pa_die("input dimensions overflow");
  return a*b;
}
static void *pa_array(size_t count,size_t width) {
  size_t bytes=pa_product(count,width);
  void *p=malloc(bytes ? bytes : 1);
  if(!p) pa_die("allocation failed");
  return p;
}
static double pa_number(const char *text) {
  char *end=NULL; errno=0;
  double value=strtod(text,&end);
  if(end==text || *end || !isfinite(value))
    pa_die("expected a finite binary64 number");
  return value;
}
static char *pa_line(FILE *file) {
  size_t n=0,capacity=128;
  char *line=(char *)pa_array(capacity,1); int c;
  while((c=fgetc(file))!=EOF && c!='\n') {
    if(n+1==capacity) {
      if(capacity>SIZE_MAX/2) pa_die("input line size overflow");
      capacity*=2; char *next=(char *)realloc(line,capacity);
      if(!next) pa_die("allocation failed");
      line=next;
    }
    line[n++]=(char)c;
  }
  if(c==EOF && n==0) { free(line); return NULL; }
  line[n]='\0'; return line;
}
typedef struct { size_t count; double *momenta; } pa_points;
static pa_points pa_read_points(const char *process,size_t external_count) {
  FILE *file=fopen("API/validation_points.dat","r");
  if(!file) pa_die("cannot open API/validation_points.dat");
  char *line=pa_line(file);
  if(!line || strcmp(line,"RUSTICOL_VALIDATION_POINTS_V1")) pa_die("invalid point-file header");
  free(line); pa_points result={0,NULL};
  size_t width=pa_product(external_count,4);
  while((line=pa_line(file))!=NULL) {
    char *token=strtok(line," \t\r");
    if(!token || strcmp(token,process)) { free(line); continue; }
    token=strtok(NULL," \t\r");
    if(!token || !*token || token[0]=='-') pa_die("invalid external count");
    char *end=NULL; errno=0; uintmax_t legs=strtoumax(token,&end,10);
    if(errno || *end || legs!=external_count) pa_die("validation external count differs");
    if(result.count==SIZE_MAX) pa_die("point count overflow");
    size_t values=pa_product(result.count+1,width);
    double *next=(double *)realloc(result.momenta,pa_product(values,sizeof(double)));
    if(!next && values) pa_die("allocation failed");
    result.momenta=next;
    for(size_t k=0;k<width;k++) {
      token=strtok(NULL," \t\r"); if(!token) pa_die("missing momentum component");
      result.momenta[result.count*width+k]=pa_number(token);
    }
    if(strtok(NULL," \t\r")) pa_die("extra momentum component");
    result.count++; free(line);
  }
  fclose(file);
  if(!result.count) pa_die("no validation points for the selected process");
  return result;
}
static void pa_values(const char *name,const double *values,size_t count) {
  printf("%s",name); for(size_t i=0;i<count;i++) printf(" %.17g",values[i]); puts("");
}
static void pa_ids(const char *name,const uint64_t *ids,size_t count) {
  printf("%s",name); for(size_t i=0;i<count;i++) printf(" %" PRIu64,ids[i]); puts("");
}
static int pa_args(int argc,char **argv,double *scale) {
  if(argc<3 || argc>4 || (argc==4 && strcmp(argv[3],"resolved")))
    pa_die("usage: check_standalone PROCESS MU_R2 [resolved]");
  *scale=pa_number(argv[2]); if(*scale<=0) pa_die("MU_R2 must be positive");
  return argc==4;
}
#endif
