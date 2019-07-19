'''
A decimal round function for python

Because the original python function used the ROUND_HALF_EVEN method, which will round the 2.5 to 2,
the following function solved this problem and round the 2.5 to 3.

python十进制下的精确四舍五入：
  由于python自带的函数默认采用“四舍六入五成双”的方法，2.5会被四舍五入成2。
  所以这个函数解决了这个问题，把2.5四舍五入成3。
'''

import math
from decimal import Decimal, ROUND_HALF_UP

def round_up(value, dec=0):
  if math.isnan(value):
    return math.nan
  else:
    multiplier = 10**dec
    value_dec = Decimal(str(value))*multiplier
    return int(value_dec.quantize(Decimal('0'),rounding=ROUND_HALF_UP))/multiplier
 
