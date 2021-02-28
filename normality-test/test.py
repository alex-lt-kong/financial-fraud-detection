#!/usr/bin/python3

import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from scipy import stats

dataset = [1.36115,
0.71242,
0.76925,
1.55454,
0,
1.15981,
3.69803,
1.28127,
1.20779,
0.90337,
0,
2.17612,
3.93554,
1.40968,
3.67842,
1.48301,
10.36871,
1.07697,
0.78045,
0.25161,
0,
1.63406,
1.73098,
12.79186,
0.9443,
1.56424,
1.31892,
5.88797,
2.22617,
8.16447,
0.44214,
3.18443,
2.24391,
0.41906,
1.7555,
1.01706,
1.29464,
0.76263,
1.90047,
0.45757,
1.03693,
0.53607,
2.57935,
1.27168,
1.75336,
13.8052,
0.8026,
0.73021,
4.53712,
2.55929,
2.58332,
3.65115,
1.5517,
1.28164,
5.29081,
0.08012,
1.88269,
0.0384,
8.99832,
2.46676,
0,
0.00179,
0.30256,
1.1651,
22.92474,
6.33795,
1.4854,
2.59718,
19.53905,
8.8079,
0.39415,
2.76642,
0.97038,
6.79908,
1.4174,
0.75137,
2.48741,
0.70565,
0.70954,
2.75818,
1.07697,
37.59164,
0.26914,
0.29666,
0.25949,
1.85542,
10.61617,
2.28882,
1.25426,
1.18216,
0.85623,
1.19487,
1.48028,
1.31626,
0.90271,
3.2272,
1.27619,
0.93578,
2.47873,
1.32697,
1.20484,
1.84367,
17.32494,
0.95885,
2.5902,
0.35766,
2.25392,
3.06113,
0.59878,
2.97294,
1.30931,
1.60837,
1.80483,
2.22129,
0.61319,
1.7308,
2.72547,
2.60621,
1.0235,
0.88373,
16.27212,
2.44803,
0.8209,
1.16554,
1.35594,
4.06875,
0.75451,
3.82344,
1.26601,
2.24865,
1.41463,
4.2146,
1.1168,
0.77574,
0.80595,
1.60469,
1.15267,
1.25077,
2.04343,
0.7799,
2.55084,
3.71141,
2.09171,
7.38889,
0.92836,
0.14815,
1.58602,
2.57494,
3.88544,
14.51064,
1.50548,
0.27273,
1.2949,
0.14047,
1.11383,
10.84321,
0.36079,
1.57272,
1.18742,
4.38645,
3.23208,
2.2991,
2.91505,
1.03066,
1.65996,
0.4593,
1.87261,
1.90261,
1.09732,
4.09534,
1.16618,
3.21785,
1.05547,
1.46617,
5.48951,
0.99929,
1.55964,
2.04103,
1.94274,
1.44289,
2.62255,
3.62396,
1.3033,
1.2826,
8.15219,
0.41485,
1.24712,
1.21717,
0.45126,
1.05934,
7.67024,
2.23102,
1.13026,
1.56083,
14.81579,
0.83547,
1.51352,
1.25488,
6.6084,
1.61686,
2.56652,
3.17722,
1.24698,
4.04996,
1.42971,
0.56365,
0.91844,
2.74866,
4.08197,
1.62855,
0.45395,
2.04344,
1.46845,
1.50173,
0.59745,
2.81129,
1.87565,
0.18809,
3.13694,
8.85646,
2.29286,
1.63922,
21.90528,
2.21424,
3.30695,
1.1527,
1.92076,
2.00219,
3.84506,
0.66807,
5.0678,
2.48154,
1.09203,
0.68301,
1.05524,
1.54204,
0.5635,
0.14281,
4.82278,
5.59371,
1.22448,
2.03828,
2.26328,
1.2013,
2.15708,
2.61222,
0.49301,
1.14055,
1.48741,
2.63181,
0,
2.12275,
0.01689,
2.44526,
6.99116,
1.64051,
1.94734,
1.69319,
1.04427,
0.29596,
1.36517,
4.99773,
1.01573,
6.30081,
119.6105,
0.67514,
1.64269,
1.07418,
7.44897,
1.73352,
6.86959,
1.44721,
0.49229,
2.70077,
0.47983,
4.96177,
1.68305,
8.47988,
1.21369,
1.74781,
1.30732,
0.71665,
2.01054,
2.01358,
0.89612,
0.49769,
0.07754,
2.37563,
2.08497,
2.72785,
1.23194,
1.77864,
0.88201,
1.23872,
1.60481,
1.13005,
4.16889,
1.73115,
1.9838,
0.89863,
0.92902,
2.85242,
3.30376,
2.78682,
1.11806,
1.86019,
1.17824,
1.2155,
1.99314,
0.95024,
2.36485,
4.89318,
1.21337,
1.44119,
7.91213,
0.28143,
0.54129,
1.70626,
5.42959,
0.93126,
0.74414,
0.80065,
2.06609,
0.11622,
5.01929,
0.58076,
2.60284,
3.13218,
42.47517,
0.39514,
1.22903,
0.47092,
1.55758,
1.09377,
0.91247,
1.68391,
2.99981,
1.18051,
3.12984,
1.9369,
1.19546,
3.41732,
10.14733,
0.71974,
20.67584,
2.75516,
1.42573,
2.28353,
0,
1.19166,
0.49676,
1.36381,
4.33745,
0.3365,
1.66347,
1.01364,
0.80064,
1.72158,
0.77355,
1.0597,
1.10504,
5.70485,
1.0499,
0.54428,
0.96366,
3.98504,
6.18119,
1.6421,
21.44051,
8.91649,
0.59804,
2.96285,
2.15222,
1.32023,
0.39044,
1.47921,
1.75897,
0.78602,
3.26819,
2.9111,
1.43006,
0,
1.9499,
1.78307,
1.72711,
2.5991,
2.31494,
2.34329,
1.61004,
1.35411,
27.7654,
2.52766,
2.15676,
1.35269,
0.72378,
6.39116,
0.64007,
0.84305,
0.41351,
0.98403,
0,
9.54846,
1.43127,
1.2477,
0.84175,
4.96095,
0.41015,
4.09534,
2.05526,
1.80702,
3.8958,
2.78893,
1.16848,
1.09346,
3.64736,
1.08765,
1.98097,
1.22138,
0.53343,
9.27097,
1.55717,
2.42062,
2.30454,
1.08794,
1.31564,
0.75629,
2.28865,
4.67834,
7.38549,
2.62259,
1.13492,
1.42551,
1.364,
3.77256,
0.71841,
1.60182,
1.78603,
5.30134,
5.19494,
2.11803,
2.8365,
1.38545,
1.17506,
0.61051,
4.31591,
1.0531,
1.41481,
4.15151,
1.06317,
1.40028,
0.78882,
7.45996,
1.96678,
2.10827,
1.1709,
1.06587,
3.019,
0.2779,
1.03503,
1.26984,
1.22644,
1.13295,
0.93025,
1.44495,
1.67659,
1.65121,
3.98213,
0.878,
1.21928,
2.54805,
1.85394,
1.26625,
1.15964,
2.60485,
1.18923,
1.66122,
1.77338,
1.1829,
6.29184,
1.82703,
1.44528,
1.87927,
4.3554,
4.58258,
13.16005,
10.4657,
1.09884,
5.90875,
2.51385,
1.88,
1.58689,
0.71305,
1.07681,
0.6354,
1.37468,
1.59112,
0.70939,
1.2698,
3.85696,
0.35722,
1.37249,
0.49553,
0.27475,
0.52377,
2.37357,
1.50344,
1.78142,
3.99367,
112.0526,
1.40902,
11.14569,
1.65358,
1.68089,
3.60596,
25.15135,
3.41785,
1.23744,
1.75496,
1.15405,
1.25179,
2.39791,
1.86969,
1.10406,
0.78664,
1.07641,
1.20582,
2.83441,
1.21062,
1.30476,
1.09018,
1.15026,
1.06147,
2.05979,
4.66717,
5.94603,
3.07607,
1.00895,
1.29279,
10.33887,
18.63058,
5.26691,
2.40495,
0.54664,
0,
1.18082,
2.84051,
22.22029,
3.42972,
1.60993,
1.27952,
1.45158,
2.23851,
1.54802,
5.4362,
1.22369,
2.60902,
2.9419,
0.57777,
2.40527,
1.80375,
0.15539,
0.84574,
4.25785,
0.21343,
0.57467,
3.25974,
1.04222,
3.43292,
1.26749,
2.14764,
0.57531,
1.00931,
0,
1.04851,
1.11889,
100.3076,
2.86012,
3.26874,
1.56132,
0.25194,
1.36253,
1.53767,
1.20944,
2.04095,
2.32889,
2.6014,
1.27014,
7.97857,
0.72471,
1.52293,
4.22491,
1.5398,
3.74594,
2.30669,
0.6739,
1.84216,
2.1709,
12.33036,
3.52514,
0.55601,
1.58275,
0.47985,
1.97038,
3.36515,
13.13808,
1.17649,
1.14949,
1.05751,
1.92963,
0.32618,
0.84404,
1.18899,
0.35101,
0.61366,
1.32166,
1.4322,
3.90494,
1.58358,
0.4199,
1.14563,
4.58864,
2.6776,
3.65276,
2.98314,
0.97702,
4.37587,
26.47223,
2.67441,
1.5466,
1.98684,
2.20745,
1.12654,
1.2435,
0.27651,
0.57746,
3.05341,
0,
0.74135,
2.94165,
12.24508,
0.17249,
8.41295,
1.63996,
1.97379,
1.23618,
2.77158,
2.19277,
1.28441,
1.44011,
0.31828,
1.97672,
1.02965,
0.06846,
0.78879,
1.06279,
0.73597,
1.89075,
0.4074,
1.18753,
1.24104,
1.37881,
4.20142,
2.14328,
318.2486,
19.39209,
36.73311,
1.1909,
2.06142,
0.85199,
1.27695,
2.14864,
0.04627,
1.72882,
1.30487,
0.43776,
0.31937,
4.42112,
1.18267,
1.8306,
1.50933,
3.77545,
2.94452,
1.49751,
7.9398,
0.6107,
22.61054,
6.82098,
0.64235,
1.03457,
0.14065,
5.87996,
1.58344,
15.97189,
1.37527,
1.05518,
6.72676,
0.60871,
1.05523,
2.44807,
1.15289,
26.0158,
1.65518,
2.00438,
0.59572,
1.4016,
0.10944,
0.99003,
5.81834,
0.96718,
0.68408,
1.11954,
1.05411,
0.71823,
0.51281,
2.77953,
1.2787,
1.31572,
2.28617,
0.73561,
1.01254,
1.26614,
0.67678,
1.01978,
0.53102,
2.51655,
0.42953,
1.57068,
1.71596,
1.29598,
1.07723,
1.29889,
0.70593,
3.58457,
2.12985,
2.92229,
1.17435,
3.08286,
1.02167,
0.25752,
1.92832,
0.124,
2.29522,
1.22366,
6.13564,
0.75576,
0.48195,
1.56525,
1.36766,
6.0781,
1.84793,
1.9447,
1.34864,
1.16558,
2.3762,
2.25696,
1.64943,
0.4712,
1.17548,
5.36843,
1.91025,
10.05305,
1.59791,
1.03814,
1.15389,
0.68812,
0.59268,
1.23053,
2.09197,
5.61271,
3.76543,
0.0112,
0.42919,
0.54751,
10.92733,
1.09348,
1.0985,
1.33547,
1.10146,
4.14352,
2.19351,
91.43377,
3.54514,
1.52269,
0.52895,
0.54009,
1.02328,
1.1767,
2.22182,
7.0541,
2.29243,
0.43194,
2.04976,
3.5022,
4.9545,
7.93437,
0.70705,
1.24274,
1.08498,
0.43361,
3.66124,
1.80697,
4.8923,
0,
1.14673,
1.32284,
0,
1.82183,
3.01388,
1.58645,
1.01287,
2.32536,
3.57409,
4.25084,
0.70787,
0.39162,
1.48317,
0,
0.95776,
1.78962,
0.18318,
1.42412,
1.18635,
2.49169,
1.07783,
1.01073,
1.75157,
15.41919,
0.64852,
2.14459,
1.2097,
0.85028,
4.84238,
7.97512,
1.23321,
0.36208,
1.09584,
1.39343,
0.41783,
0.80546,
1.2992,
1.63963,
1.30469,
0.90131,
0.27746,
0,
1.47015,
4.91362,
1.09757,
1.04204,
1.72829,
0.72527,
3.63143,
5.17868,
0.3397,
1.82504,
3.45512,
1.81877,
0.1903,
0.50179,
4.03439,
2.60848,
1.46524,
2.46278,
1.71784,
2.58491,
2.3277,
4.11264,
0.49732,
1.0345,
1.88718,
0.58255,
2.82569,
1.52733,
2.72964,
1.43143,
1.45043,
2.03292,
1.04705,
1.27064,
4.44003,
1.04631,
1.42326,
0.6102,
2.1159,
18.20049,
0.70929,
1.54336,
0.17529,
1.58463,
3.00699,
1.91983,
6.73763,
2.48972,
0.30259,
6.55274,
1.21062,
1.35164,
2.76795,
0.30766,
0.0991,
1.30678,
0.39507,
1.40331,
0.00329,
1.56143,
0.36089,
1.80786,
0.96033,
5.25965,
0.492,
1.74096,
4.02127,
1.43836,
1.32426,
1.68086,
1.56995,
0.75349,
0.76488,
2.19616,
0.359,
0.76868,
0.67723,
1.66062,
1.28906,
1.43825,
0.0064,
2.32806,
0.74964,
135.152,
3.64982,
1.44625,
1.1267,
0.57286,
1.32994,
1.2735,
0,
1.6227,
3.82883,
1.29107,
1.42271,
2.85173,
0.45037,
1.36076,
1.66898,
1.60702,
0.53961,
3.20864,
3.02009,
2.75596,
1.82641,
2.22931,
2.46468,
1.47775,
1.2065,
2.54641,
1.10584,
1.67439,
1.23921,
2.70731,
0.56079,
0.56102,
0.32145,
3.8921,
1.18612,
0.01139,
1.93731,
0.83219,
3.82175,
1.81171,
1.10056,
2.13079,
1.07569,
1.45204,
2.37641,
1.32818,
1.5636,
1.18537,
1.64824,
4.13887,
0.79191,
10.44306,
1.21424,
4.63659,
0.62466,
1.64602,
2.84035,
1.86891,
2.85822,
0.93951,
1.30404,
0.71839,
0.97645,
1.49831,
3.06697,
4.3746,
3.21668,
1.25764,
1.53758,
3.35636,
1.17884,
0.68286,
1.09701,
3.67651,
11.91334,
1.21348,
0.87216,
0.08164,
0.77047,
0.17331,
2.23694,
3.75463,
1.87554,
1.25579,
1.0602,
1.25171,
6.94615,
4.07824,
1.43402,
1.79898,
1.27356,
0.70455,
0.22358,
0.99017,
2.15413,
1.63638,
19.79838,
0,
32.6279,
2.52321,
0.22402,
3.23103,
37.20969,
1.5393,
1.78923,
4.52851,
10.84788,
15.15035,
0.35435,
1.95812,
1.91167,
1.18258,
1.35309,
1.60514,
3.7751,
0.98737,
5.09865,
1.29633,
2.69229,
1.84022,
7.88914,
2.49701,
1.43948,
0.19758,
2.56857,
1.22347,
1.09554,
1.6336,
1.19179,
1.00319,
2.17246,
1.48681,
1.79618,
2.75433,
1.6235,
0.7183,
1.07984,
2.34325,
0.41815,
1.19016,
0.8083,
0.22861,
1.44592,
1.17819,
1.78842,
1.79201,
1.23103,
3.61243,
0.32215,
2.53397,
1.31119,
4.98415,
1.28969,
1.77415,
0,
5.13892,
2.74115,
1.35724,
1.12928,
1.12852,
6.00556,
11.06577,
0,
2.60781,
2.62565,
1.77538,
7.78394,
1.43672,
1.30842,
1.00115,
1.26335,
1.65522,
0.81452,
2.09272,
1.75768,
1.5099,
3.60362,
1.5878,
2.07802,
1.76472,
0.88227,
33.21848,
2.25919,
1.24619,
0.35028,
1.09968,
1.04634,
2.47968,
0,
2.10969,
1.49886,
0,
2.07047,
3.26169,
1.46812,
3.09194,
2.05759,
3.14851,
1.88065,
1.99219,
1.85248,
0.37168,
4.56878,
0.5576,
0,
4.30878,
3.79488,
2.83319,
0.67292,
0.20558,
1.54956,
0.60258,
2.52397,
0.53584,
1.21572,
0.6495,
0.28303,
1.41578,
1.38446,
1.30226,
1.42666,
3.09642,
1.05442,
1.79795,
0.22433,
3.93159,
0.29804,
1.30852,
2.34352,
3.30321,
0.09581,
2.34526,
1.45656,
1.00776,
0,
1.1099,
0.14784,
1.16752,
1.78499,
1.5622,
1.35677,
3.08277,
3.74227,
1.42154,
1.08396,
2.43861,
3.25877,
7.84113,
1.60937,
2.07154,
1.09999,
1.38042,
1.85955,
2.77817,
0.43272,
1.38195,
0.58485,
1.14388,
0.70265,
0.62656,
1.18724,
3.48667,
0.55533,
1.09254,
4.69226,
5.0248,
3.73848,
1.33776,
2.48364,
1.29994,
1.09287,
7.51693,
2.10855,
2.43963,
8.87588,
1.03422,
4.48992,
3.48594,
1.6205,
32.00185,
0.97615,
1.58128,
1.04807,
2.1272,
0.89991,
0.54937,
2.14954,
1.28477,
0.93525,
1.77551,
2.97043,
18.97982,
1.62612,
1.03425,
0,
5.19719,
2.43182,
2.31042,
2.13662,
6.26121,
8.33705,
2.53874,
3.4525,
0.36989,
0.61554,
1.69884,
0.50254,
2.71074,
2.85171,
1.06813,
0.63554,
5.82377,
1.12136,
1.28689,
4.23265,
1.20461,
3.66222,
2.83598,
1.30586,
1.46036,
2.46786,
1.08752,
0,
1.69189,
2.19886,
0.05463,
1.57125,
1.51297,
2.30602,
2.39797,
1.17416,
1.63259,
6.66821,
0.41841,
1.02583,
1.61997,
1.00833,
1.34744,
2.21054,
6.27357,
0.53885,
0.83653,
3.33235,
14.40172,
0,
4.22724,
4.61241,
1.57435,
1.29922,
4.08028,
1.16901,
1.16254,
4.99338,
5.15699,
0.57707,
4.73708,
1.83138,
1.74087,
2.72457,
1.95261,
1.22667,
1.17668,
1.17714,
0,
1.1301,
2.30959,
1.00712,
8.7309,
2.74997,
1.63634,
2.49289,
2.50675,
1.82984,
1.1357,
2.54211,
3.30974,
1.24889,
1.46689,
1.20361,
1.83945,
1.60047,
1.27466,
1.67078,
4.00017,
1.27355,
2.18733,
0.82229,
1.26644,
2.7761,
1.53987,
1.74836,
1.5746,
1.43612,
4.53173,
1.25857,
1.5383,
3.22817,
1.46351,
1.13544,
1.70386,
2.15583,
3.80365,
0,
3.77893,
1.16808,
4.45368,
1.32979,
1.43361,
3.41633,
4.48628,
1.44365,
1.08061,
3.44177,
1.29299,
36.42329,
1.56822,
1.3592,
15.46404,
2.39767,
3.20674,
1.04341,
1.85887,
4.90512,
10.44695,
1.01173,
0.27319,
3.79578,
2.36527,
1.79248,
2.76533,
0.89904,
4.02187,
5.10239,
2.08613,
0.50293,
1.03965,
3.4397,
0.92197,
0.76998,
1.24721,
1.54049,
6.22393,
1.37256,
3.98084,
2.98599,
1.39751,
3.22587,
0.86226,
2.62759,
13.1773,
1.65094,
1.7315,
0.58466,
1.82056,
9.16531,
2.60902,
3.65724,
1.74168,
22.41677,
1.08091,
0.9348,
1.91788,
1.1578,
2.96985,
4.36868,
1.09286,
2.10086,
2.44917,
2.61494,
0.23208,
1.96571,
5.37752,
2.98605,
2.64921,
3.68626,
2.76319,
7.79077,
1.32247,
0.22974,
1.13811,
4.77004,
3.89144,
7.96361,
6.66904,
1.68886,
0.33402,
1.50354,
0.48657,
1.81467,
2.00286,
21.5441,
2.50297,
1.92559,
1.11632,
1.25504,
5.0589,
1.38572,
1.08622,
1.75431,
0.51542,
3.11462,
1.48702,
1.50329,
1.33849,
4.52552,
0.07671,
6.93934,
2.83222,
1.29203,
2.87074,
0.41604,
1.12154,
3.41,
2.01926,
1.15546,
1.87721,
2.42588,
2.56825,
0.55943,
1.13637,
1.00523,
5.22772,
1.2657,
1.7445,
1.1713,
0.25949,
3.78561,
1.35841,
1.49206,
0.63817,
0.85301,
1.18303,
8.1941,
0.92398,
1.91265,
0.9571,
1.23977,
2.44604,
0.40737,
1.76071,
0.78981,
1.66809,
1.56001,
1.13554,
2.05483,
0.7713,
6.07192,
3.18099,
3.7126,
3.95767,
3.96804,
0.51625,
1.52226,
4.77163,
1.22195,
7.04968,
2.90783,
4.57782,
1.43982,
1.57987,
3.11717,
2.17563,
0.77709,
1.31121,
1.36724,
1.56873,
1.0788,
3.0072,
2.56412,
1.85831,
1.93255,
1.26041,
0.72871,
5.28122,
0.58412,
3.61578,
3.39144,
0.87228,
3.85057,
3.37408,
1.64653,
14.99974,
4.18543,
0.46917,
3.15691,
0.35939,
0.24603,
2.16996,
1.4536,
2.18122,
6.43768,
1.10755,
2.76487,
0.82829,
2.76937,
9.33886,
2.65912,
2.35928,
1.88039,
8.90634,
0.65841,
1.20799,
1.63137,
1.4868,
1.2652,
1.62332,
1.10752,
1.02843,
0.47129,
1.83719,
20.74341,
1.46915,
2.47764,
1.96862,
1.37812,
0,
0,
9.40267,
1.16718,
1.0231,
6.16423,
1.22833,
1.11984,
2.89363,
1.60537,
1.54955,
2.40456,
2.48602,
1.46056,
3.11533,
2.43684,
7.48153,
1.70081,
1.03594,
2.24447,
1.38634,
2.20398,
5.84512,
2.03158,
2.08728,
7.8545,
1.55876,
1.56567,
5.67627,
0.92544,
1.17786,
1.7094,
2.86189,
1.10596,
0,
1.13751,
3.94536,
2.74464,
0.50787,
3.52949,
1.57104,
1.33361,
6.53156,
1.85261,
1.08208,
1.16528,
3.8204,
3.03653,
1.56826,
0,
1.71322,
1.14812,
5.08823,
0,
0.10204,
1.32578,
0.84489,
0.6992,
0.97466,
1.52289,
1.30632,
0.33239,
0.00508,
1.10677]

#da = pd.read_csv('/home/mamsds/Documents/d/hkust/courses/mfit5005_foundations-of-fintech/group-projects/normality-test/geDJ_with-headers.txt', sep='\s+', header = 0)
#da['date'] = pd.to_datetime(da['date'], format='%Y%m%d')
#da['log_result'] = np.log(da['close']) 
#rt = np.diff(da['log_result'])
#rt = np.append(np.nan, rt)
# The NaN value 'infects' all other statistics that follow. 
#da['log_return'] = rt

#print(da["log_return"])
#print('skew: {}'.format(da["log_return"].skew()))
#print('kurtosis: {}'.format(da["log_return"].kurtosis()))
#statistic, pvalue = stats.normaltest(da["log_return"][1:])
#print('statistic: {}, pvalue: {}'.format(statistic, pvalue))

#da['log_return'][:].plot.kde(grid = True)

#plt.show()
newda = pd.DataFrame(dataset)
statistic, pvalue = stats.normaltest(newda)
pvalue = float(pvalue)
print('statistic: {}, pvalue: {}'.format(statistic, pvalue))
# As a rule of thumb, if skewness and kurtosis are both between -1 and 1, the distribution is likely to be normal.
skewness = float(newda.skew())
kurtosist = float(newda.kurtosis())
mu, std = stats.norm.fit(newda) # mu means the real mean of all data. σ (sigma) just means standard deviation
sigma = math.sqrt(std)

newda.plot.kde(grid = True)
plt.title(r'A normal distribution')
plt.legend(['mu: {}, σ: {}\np value:{}\nskewness: {}\nkurtosis: {}'.format(round(mu, 3), round(sigma, 3), round(pvalue, 3), round(skewness, 3), round(kurtosist, 3))])
plt.axvline(mu, color='red')
plt.axvline(mu + std, color='pink')
plt.text(mu, 0, 'mu:{}'.format(round(mu, 3)),rotation=90)
plt.axvline(mu + std * 2, color='pink')
plt.axvline(mu + std * 3, color='pink')
plt.axvline(mu - std, color='pink')
plt.axvline(mu - std * 2, color='pink')
plt.axvline(mu - std * 3, color='pink')
#fig, ax = plt.subplots(figsize=(9,6))
#ax.fill_between(mu-std,mu+std,0, alpha=0.3, color='b')
#ax.fill_between(x_all,y2,0, alpha=0.1)
plt.show()

# according to this link: https://stackoverflow.com/questions/50276138/how-is-pandas-kurtosis-defined
# kurtosis of pandas is defined as kurtosis - 3.
