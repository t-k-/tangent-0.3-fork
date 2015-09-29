from tangent.math.mathdocument import MathDocument 
from tangent.utility.control import Control
from sys import argv

cntl = Control('./tangent.cntl')
d = MathDocument(cntl)
print(d.find_doc_file(int(argv[1])))  # doc_num and pos_num
print(d.find_mathml(int(argv[1]), int(argv[2])))  # doc_num and pos_num
