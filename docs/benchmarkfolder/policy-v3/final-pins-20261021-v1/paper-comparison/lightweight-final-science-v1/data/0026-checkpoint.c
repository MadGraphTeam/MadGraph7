/* Private byte-copy checkpoint. No numerical arguments, arithmetic or engines. */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static void fail(void) { _Exit(95); }

void dpstab_custody_checkpoint(int phase) {
    const char *root=getenv("DPSTAB_CUSTODY_DIR");
    const char *prefix=getenv("DPSTAB_PROCESS_PREFIX");
    const char *logname=getenv("DPSTAB_CUSTODY_LOG");
    const char *stage=(phase==1 ? "postwarm" : phase==2 ? "final" : NULL);
    const char *files[2]={"HelFilter.dat","LoopFilter.dat"};
    if (!root || root[0]!='/' || !prefix || !prefix[0] || !logname || !stage ||
        strlen(prefix)>63 || strspn(prefix,"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")!=strlen(prefix)) fail();
    FILE *log=fopen(logname,"a");
    if (!log) fail();
    for (int i=0;i<2;i++) {
        char source[4096],target[4096];
        int ns=snprintf(source,sizeof(source),"MadLoop5_resources/%s%s",prefix,files[i]);
        int nt=snprintf(target,sizeof(target),"%s/%s/%s%s",root,stage,prefix,files[i]);
        if (ns<0 || nt<0 || (size_t)ns>=sizeof(source) || (size_t)nt>=sizeof(target)) fail();
        struct stat existing;
        if (!lstat(target,&existing) || errno!=ENOENT) fail();
        FILE *input=fopen(source,"rb");
        if (!input) {
            if (errno!=ENOENT || fprintf(log,"SNAPSHOT phase=%d stage=%s file=%s available=0 bytes=0\n",phase,stage,files[i])<0) fail();
            continue;
        }
        FILE *output=fopen(target,"wbx");
        if (!output) fail();
        unsigned char buffer[8192];
        size_t total=0,amount;
        while ((amount=fread(buffer,1,sizeof(buffer),input))!=0) {
            if (fwrite(buffer,1,amount,output)!=amount) fail();
            total+=amount;
        }
        if (ferror(input) || fclose(input) || fclose(output)) fail();
        if (fprintf(log,"SNAPSHOT phase=%d stage=%s file=%s available=1 bytes=%zu\n",phase,stage,files[i],total)<0) fail();
    }
    if (fclose(log)) fail();
}
