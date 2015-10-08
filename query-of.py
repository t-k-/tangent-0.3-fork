import re
import sys

query_name = sys.argv[1]

fh = open('testing/test_queries/tk-query.xml')
re_res = re.findall(r'<m:annotation encoding="application/x-tex".*?>(.*?)</m:annotation>', fh.read(), re.DOTALL)

for (i, tex) in enumerate(re_res):
    if query_name == 'NTCIR11-Math-%d.txt' % (i + 1):
        oneline_tex = tex.replace('\n', '')
        # print(str(i + 1) + ": " + oneline_tex)
        print(oneline_tex)
