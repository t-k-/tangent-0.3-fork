import codecs
import sys
from sys import argv

from tangent.utility.control import Control
from tangent.math.mathdocument import MathDocument

__author__ = 'FWTompa'



if __name__ == '__main__':

    if sys.stdout.encoding != 'utf8':
      sys.stdout = codecs.getwriter('utf8')(sys.stdout.buffer, 'strict')
    if sys.stderr.encoding != 'utf8':
      sys.stderr = codecs.getwriter('utf8')(sys.stderr.buffer, 'strict')

    if len(argv) != 4 or argv[1] == "help":
        print("Use: python get_math.py <cntl> <doc#> <expr#>")
        print("        where (doc# < 0) => use queryfile")
        sys.exit()

    cntl = Control(argv[1]) # control file name (after indexing)
    d = MathDocument(cntl)
    print(d.find_mathml(int(argv[2]),int(argv[3])))  # doc_num and pos_num
