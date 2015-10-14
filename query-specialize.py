import re
import sys
import html

query = sys.argv[1]
query = re.sub(r'\\qvar{([a-z]+)([0-9]+)}', r'\\varUpsilon', query)
query = html.unescape(query)
print(query)
