"""
The purpose of this module is to address the reviewer 1 comment
from gh-43 requesting human readable justifications for the
viral host target (re)labels.
"""

from collections import defaultdict
import pandas as pd

justification_dict = defaultdict(dict)
with open("retarget.py") as infile:
    started = False
    justification_found = False
    lines = infile.readlines()
    counter = 0
    for idx, line in enumerate(lines):
        stripped_line = line.strip()
        if stripped_line.startswith("organism_dict"):
            started = True
        if started and stripped_line.startswith("# http") and not justification_found:
            justification_dict[counter]["justification"] = stripped_line[2:]
            justification_found = True
        if started and justification_found:
            if stripped_line.count('"') == 4:
                virus_name, host_label = stripped_line.split(":")
                host_label = host_label.replace(",", "")
                justification_dict[counter]["virus name"] = virus_name
                justification_dict[counter]["host label"] = host_label
                counter += 1
                justification_found = False

df = pd.DataFrame.from_dict(justification_dict,
                            orient="index",
                            columns=["virus name", "host label", "justification"])
print(df)
html = df.to_html(render_links=True)
with open("table.html", "w") as outfile:
    outfile.writelines(html)


