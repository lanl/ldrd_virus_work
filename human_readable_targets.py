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
    lines = infile.readlines()
    counter = 0
    line_idx = 0
    while line_idx < 3151:
        stripped_line = lines[line_idx].strip()
        if stripped_line.startswith("organism_dict"):
            started = True
        if started and stripped_line.startswith("#"):
            comment_block = ""
            for line_index in range(line_idx, len(lines)):
                stripped_sub_line = lines[line_index].strip()
                if stripped_sub_line.startswith("#"):
                    stripped_sub_line += "<br>"
                    comment_block += stripped_sub_line
                    line_idx += 1
                else:
                    break
            justification_dict[counter]["justification"] = comment_block
            continue
        if started and stripped_line.count('"') == 4:
            virus_name, host_label = stripped_line.split(":")
            host_label = host_label.replace(",", "")
            justification_dict[counter]["virus name"] = virus_name
            justification_dict[counter]["host label"] = host_label
            counter += 1
        line_idx += 1

df = pd.DataFrame.from_dict(justification_dict,
                            orient="index",
                            columns=["virus name", "host label", "justification"])
print(df)
html = df.to_html(render_links=True, escape=False, justify='left')
with open("table.html", "w") as outfile:
    outfile.writelines(html)


