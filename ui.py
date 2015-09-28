from tangent.math.mathdocument import MathDocument 
from tangent.utility.control import Control
from sys import argv

cntl = Control('./tangent.cntl')
d = MathDocument(cntl)
print(d.find_mathml(int(argv[1]), 0))  # doc_num and pos_num
