// Copyright (C) 2013-2017 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

 // ACL kernel for converting from conserved to primitive variables
__kernel void vector_add(__global const float *cons_, 
                         __global float *restrict prim_, 
                         __read_only const unsigned size,
                         __read_only const float density_floor_,
                         __read_only const float pressure_floor_,
                         __read_only const float gm1_)
{
    // get index of the work item
    int index = get_global_id(0);


    const unsigned IDN = 0;
    const unsigned IM1 = 1;
    const unsigned IM2 = 2;
    const unsigned IM3 = 3;
    const unsigned IEN = 4;

    // const unsigned IDN = 0;
    const unsigned IVX = 1;
    const unsigned IVY = 2;
    const unsigned IVZ = 3;
    const unsigned IPR = 4;
    
    float& u_d  = cons_[IDN*size + index];
    float& u_m1 = cons_[IM1*size + index];
    float& u_m2 = cons_[IM2*size + index];
    float& u_m3 = cons_[IM3*size + index];
    float& u_e  = cons_[IEN*size + index];

    float& w_d  = prim_[IDN*size + index];
    float& w_vx = prim_[IVX*size + index];
    float& w_vy = prim_[IVY*size + index];
    float& w_vz = prim_[IVZ*size + index];
    float& w_p  = prim_[IPR*size + index];

    // apply density floor, without changing momentum or energy
    u_d = (u_d > density_floor_) ?  u_d : density_floor_;
    w_d = u_d;

    float di = 1.0/u_d;
    w_vx = u_m1*di;
    w_vy = u_m2*di;
    w_vz = u_m3*di;

    float ke = 0.5*di*(u_m1*u_m1 + u_m2*u_m2 + u_m3*u_m3);
    w_p = gm1_*(u_e - ke);

    // apply pressure floor, correct total energy
    u_e = (w_p > pressure_floor_) ?  u_e : ((pressure_floor_/gm1_) + ke);
    w_p = (w_p > pressure_floor_) ?  w_p : pressure_floor_;

}

