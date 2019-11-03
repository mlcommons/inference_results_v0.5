#pragma once
inline void pad(char* a, char* b, int w, int c1, int c2) 
{
  for(int i = 0; i < w;i++) {
    for(int j = 0; j < c1; j ++)
      *b++ = *a++;
    for(int j = 0; j < c2-c1;j++)
      *b++ = 0;
  }
}

inline void transpose(char* a, char* b, int ho, int hi, int w, int c)
{
  int idx = 0;
  for(int i = 0; i < ho; i ++) {
    for(int l = 0; l < c; l ++) {
      for(int j = 0; j < hi; j ++) {
        for(int k = 0; k < w; k ++) {
          *b++ = a[idx]; // i*j*w*c+k*c+l];
          idx += c;
        }
      }
      idx += 1 - hi * w * c;
    }
    idx += hi * w * c - c;
  }
}
