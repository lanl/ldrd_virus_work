import re


from Bio import SeqIO
from Bio.SeqUtils import seq1, seq3
import MDAnalysis as mda
from ggmolvis.ggmolvis import GGMolVis
import bpy


ggmv = GGMolVis()

# kmer_PC_65606562 matches the following in
# Adam's latest results for sialic acid target:
candidate_1 = "LRLGMRIS"
candidate_2 = "LRLGMRVS"
candidate_3 = "LRLGIKLT"


# PDB: 9cyx is for Mammalian orthoreovirus 3 Dearing,
# which is one of the hits for kmer_PC_65606562

# do some data munging with MDAnalysis to see if
# if there is a match for one of the above candidate AA
# seqs in PDB file, and if so, find the index of the match
# in the PDB file/MDA Universe:
u = mda.Universe("9cyx.pdb")
protein = u.select_atoms("protein")
residues = protein.residues
resnames = seq1("".join(residues.resnames))
sels_of_interest = []
for candidate in (candidate_1, candidate_2, candidate_3):
    pattern = re.compile(re.escape(candidate), re.IGNORECASE)
    matches = pattern.finditer(resnames)
    positions = [match.start() for match in matches]
    if positions:
        print(f"'{candidate}' found at the following 0-based residue positions:")
        for pos in positions:
            print(f"Position {pos}")
            sels_of_interest.append([pos, pos + len(candidate)])

for start_idx, end_idx in sels_of_interest:
    print("residue start_idx, end_idx:", start_idx, end_idx)
    print("selection of interest:", residues[start_idx:end_idx])

# for now, manually encode the single match from above
kmer_match = residues[2492:2500]
# kmer representation:
kmer_mol = ggmv.molecule(kmer_match, lens=80, material="default", color="red")
# lambda 1 representation:
lambda_1 = protein.select_atoms("chainID H I B")
ggmv.molecule(lambda_1, lens=80, material="default", color="blue")
# lambda 2 representation:
lambda_2 = protein.select_atoms("chainID A")
ggmv.molecule(lambda_2, lens=80, material="default", color="green")
# sigma 2 representation:
sigma_2 = protein.select_atoms("chainID Q R")
ggmv.molecule(sigma_2, lens=80, material="default", color="violet")
# protein representation:
protein_mol = ggmv.molecule(protein, lens=80, material="default")

# turn on METAL for rendering (for Mac ARM)
bpy.context.scene.render.engine = "CYCLES"
prefs = bpy.context.preferences
cycles_prefs = prefs.addons['cycles'].preferences
cycles_prefs.get_devices()
cycles_prefs.compute_device_type = 'METAL'
for device in cycles_prefs.devices:
    if 'Apple' in device.name and 'GPU' in device.name:
        device.use = True

protein_mol.render(resolution=(1000, 1000),
               filepath="test.png",
               mode="image")
