import re
import sys

query = sys.argv[1]
query = re.sub(r'\\qvar{([a-z]+)([0-9]+)}', r'\\varUpsilon_\2', query)
print(query)
