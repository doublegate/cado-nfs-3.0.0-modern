#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define GF2X_WORDSIZE 64
#define GF2X_HAVE_VPCLMUL_SUPPORT 1
#define GF2X_HAVE_PCLMUL_SUPPORT 1
#define GF2X_STORAGE_CLASS_mul2 static
#define GF2X_STORAGE_CLASS_mul3 static
#define GF2X_STORAGE_CLASS_mul4 static
#define GF2X_FUNC(x) x
#include "gf2x_mul2.h"
#include "gf2x_mul3.h"
#include "gf2x_mul4.h"
static void clmul64(uint64_t a,uint64_t b,uint64_t*lo,uint64_t*hi){uint64_t l=0,h=0;for(int i=0;i<64;i++)if((b>>i)&1){l^=a<<i;if(i)h^=a>>(64-i);}*lo=l;*hi=h;}
static void ref(uint64_t*c,const uint64_t*a,const uint64_t*b,int N){for(int i=0;i<2*N;i++)c[i]=0;for(int i=0;i<N;i++)for(int j=0;j<N;j++){uint64_t lo,hi;clmul64(a[i],b[j],&lo,&hi);c[i+j]^=lo;c[i+j+1]^=hi;}}
static uint64_t rng(uint64_t*s){*s^=*s<<13;*s^=*s>>7;*s^=*s<<17;return *s;}
int main(void){uint64_t s=0xABCDEF1ULL;long b2=0,b3=0,b4=0;int T=200000;
for(int t=0;t<T;t++){unsigned long a[4],b[4],cr[8],cg[8];for(int i=0;i<4;i++){a[i]=rng(&s);b[i]=rng(&s);}
ref(cr,a,b,2);gf2x_mul2(cg,a,b);if(memcmp(cr,cg,32))b2++;
ref(cr,a,b,3);gf2x_mul3(cg,a,b);if(memcmp(cr,cg,48))b3++;
ref(cr,a,b,4);gf2x_mul4(cg,a,b);if(memcmp(cr,cg,64))b4++;}
printf("integrated headers vs scalar ref (%d trials): mul2 %s(%ld) mul3 %s(%ld) mul4 %s(%ld)\n",T,b2?"FAIL":"PASS",b2,b3?"FAIL":"PASS",b3,b4?"FAIL":"PASS",b4);
printf("%s\n",(b2||b3||b4)?"FAILURES":"ALL PASS");return (b2||b3||b4)?1:0;}
