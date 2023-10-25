from pybiomart import Server
import pandas as pd

server = Server(host="http://www.ensembl.org")
mart = server["ENSEMBL_MART_ENSEMBL"]
dataset = mart["hsapiens_gene_ensembl"]

# Data from Shaw et al. PLOS Biol. 2017
# retrieved from http://isg.data.cvr.ac.uk/
df = pd.read_csv("ISG_source.csv")
ISG_transcript_ids = set()
bad_not_present: list[str] = []
batch_size = 100
genes = list(df["ENSEMBL ID"].values)
for i in range(0, len(genes), batch_size):
    result = dataset.query(
        attributes=["ensembl_transcript_id"],
        filters={
            "link_ensembl_gene_id": genes[i : i + batch_size],
            "transcript_is_canonical": True,
        },
    )
    if len(result) > 0:
        ISG_transcript_ids.update(result["Transcript stable ID"].values)

with open("ISG_transcript_ids.txt", "w") as f:
    for each in ISG_transcript_ids:
        print(each, end=" ", file=f)
