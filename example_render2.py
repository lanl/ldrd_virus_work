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
candidate_4 = "LKLGIKLT"
candidate_5 = "LQLGIALT" # from COBALT alignment to respirovirus 3


# do some data munging with MDAnalysis to see if
# if there is a match for one of the above candidate AA
# seqs in PDB file, and if so, find the index of the match
# in the PDB file/MDA Universe:
u = mda.Universe("1ztm.pdb")
protein = u.select_atoms("protein")
residues = protein.residues
resnames = seq1("".join(residues.resnames))
sels_of_interest = []
for candidate in (candidate_1, candidate_2, candidate_3, candidate_4, candidate_5):
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
kmer_match1 = residues[128:136]
kmer_match2 = residues[545:553]
kmer_match3 = residues[970:978]

# kmer representations:
ggmv.molecule(kmer_match1, lens=80, material="default", color="red")
ggmv.molecule(kmer_match2, lens=80, material="default", color="red")
ggmv.molecule(kmer_match3, lens=80, material="default", color="red")
# protein representation:
protein_mol = ggmv.molecule(protein.select_atoms("chainID A"),
                                                 lens=80,
                                                 material="default",
                                                 color="green")
protein_mol = ggmv.molecule(protein.select_atoms("chainID B"),
                                                 lens=80,
                                                 material="default",
                                                 color="blue")
protein_mol = ggmv.molecule(protein.select_atoms("chainID C"),
                                                 lens=76,
                                                 material="default",
                                                 color="violet")

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
               filepath="test2.png",
               mode="image")
