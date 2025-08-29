"""
See: https://gitlab.lanl.gov/treddy/ldrd_virus_work/-/issues/54

The purpose of this module is to retarget the original Mollentze data
from True/False on human infection to True/False on mammal and primate
targets. This module should also improve the human vs. no-human infection
labels, since many records are manually inspected during the task of
applying new host labels.

This module should be executed in the repo root, but you'll need to
manually provide the path to your local cache of the records.
"""

import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO

# dictionary mapping to known host (always specify "human" if known)
organism_dict = {
                # https://doi.org/10.1128/jvi.02364-10
                "Bat picornavirus": "non_primate_mammals",
                # https://doi.org/10.3390/vetsci7020079
                "Infectious bronchitis virus": "avian",
                # https://doi.org/10.3201/eid2012.131742
                # single study stating detection in pigs in korea
                # same study as "Gouleako virus"
                "Herbert virus strain F23/CI/2004": "non_primate_mammals",
                # https://doi.org/10.1007/s12250-017-3973-z
                "Hepatitis E virus rat/R63/DEU/2009": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/200403
                "Mosqueiro virus": "no_mammals",
                # https://www.genome.jp/virushostdb/1637527
                "Dromedary camel enterovirus 19CC": "non_primate_mammals",
                # https://doi.org/10.1371/journal.pone.0260038
                "Rat coronavirus Parker": "non_primate_mammals",
                # https://doi.org/10.1016/j.virol.2006.02.027
                "Newbury agent 1": "non_primate_mammals",
                # https://doi.org/10.3201/eid2207.160024
                # aka Porcine pegivirus
                "Pegivirus suis": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/33724
                "Nilaparvata lugens reovirus": "no_mammals",
                # https://doi.org/10.3201/eid2012.131742
                # single study stating detection in pigs in korea
                "Gouleako virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/10564
                "Deltapapillomavirus 2": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/10317
                "Cercopithecine alphaherpesvirus 2": "primate",
                # https://www.genome.jp/virushostdb/10276
                "Swinepox virus": "non_primate_mammals",
                # https://doi.org/10.1007/s00705-003-0271-x
                "Oita virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/376852
                "Goatpox virus Pellor": "non_primate_mammals",
                # https://doi.org/10.1016/j.virol.2012.08.008
                "Kimberley virus": "no_mammals",
                # https://www.genome.jp/virushostdb/943342
                "Eothenomys miletus hantavirus LX309": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/765147
                "Kenkeme virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2870348
                # aka "African bat icavirus PREDICT-06105" (https://www.uniprot.org/taxonomy/2870348)
                "mischivirus C1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2560310
                # aka "Avian metaavulavirus 11" 
                "avian paramyxovirus 11": "avian",
                # same accession as: "Maize fine streak nucleorhabdovirus":
                # https://doi.org/10.1094/PHYTO.2002.92.11.1167
                "Maize fine streak virus": "no_mammals",
                # same accession as: "Simian sapelovirus 1"
                # but spelled with a lowercase 's'
                # https://www.genome.jp/virushostdb/1002918
                "simian sapelovirus 1": "non_primate_mammals",
                # same accession as "Loveridges garter snake virus 1"
                # but with an apostrophe
                # https://doi.org/10.1128/genomea.00779-14
                "Loveridge's garter snake virus 1": "no_mammals",
                # https://www.genome.jp/virushostdb/2560318
                "avian paramyxovirus 8": "avian",
                # https://www.genome.jp/virushostdb/1708572
                "hepatovirus B1": "non_primate_mammals",
                # https://doi.org/10.1128/jvi.01094-13
                # aka Eel picornavirus 1
                "potamipivirus A1": "no_mammals",
                # https://www.genome.jp/virushostdb/1534546
                # aka  Chicken picornavirus 3
                "avisivirus C1": "avian",
                # https://www.genome.jp/virushostdb/1294177
                # aka Sebokele virus 1
                "parechovirus C1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2773293
                "rabovirus A1": "non_primate_mammals",
                # https://doi.org/10.1128/jvi.01394-12
                # aka Miniopterus schreibersii picornavirus 1
                "mischivirus A1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/64300
                "Modoc virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/3052633
                # aka Wutai Mosquito Phasivirus (https://doi.org/10.3390/insects10050135)
                "Phasivirus wutaiense": "no_mammals",
                # https://www.genome.jp/virushostdb/376849
                "Lumpy skin disease virus NI-2490": "non_primate_mammals",
                # https://doi.org/10.1007/BF01250066
                "Parana virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1002359
                "Drosophila ananassae sigmavirus": "no_mammals",
                # https://www.genome.jp/virushostdb/74581 
                "Rat parvovirus 1": "non_primate_mammals",
                # https://doi.org/10.1126/science.6170110
                "Moloney murine sarcoma virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2847812
                "Keterah virus": "no_mammals",
                # https://www.genome.jp/virushostdb/2870348
                # aka mischivirus C1 (https://www.uniprot.org/taxonomy/2870348)
                "African bat icavirus PREDICT-06105": "non_primate_mammals",
                "mischivirus C1": "non_primate_mammals",
                # https://doi.org/10.3389/fcimb.2017.00490
                "Estero Real virus": "no_mammals",
                # https://doi.org/10.1016/j.meegid.2017.01.019
                "Wuhan Fly Virus 1": "no_mammals",
                # https://www.genome.jp/virushostdb/1985698
                "Varicosavirus lactucae": "no_mammals",
                # https://www.genome.jp/virushostdb/1155188
                "Bovine rhinitis A virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/693998
                "Rhinolophus bat coronavirus HKU2": "non_primate_mammals",
                # https://doi.org/10.3390/v13060982
                # aka Okahandja mammarenavirus aka Okahandja virus
                "Mammarenavirus okahandjaense": "non_primate_mammals",
                # https://doi.org/10.1186/s12985-018-1093-5
                "Equid alphaherpesvirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/745716
                "Coffee ringspot virus": "no_mammals",
                # https://www.genome.jp/virushostdb/2560310
                # aka avian paramyxovirus 11 
                "Avian metaavulavirus 11": "avian",
                "avian paramyxovirus 11": "avian",
                # https://doi.org/10.1094/PHYTO.2002.92.11.1167
                "Maize fine streak nucleorhabdovirus": "no_mammals",
                # https://doi.org/10.1007/s007050050658
                "Rabbit picobirnavirus": "non_primate_mammals", 
                # https://doi.org/10.1016/0042-6822(85)90281-8
                "Wound tumor virus": "no_mammals",
                # https://www.genome.jp/virushostdb/10497
                "African swine fever virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/273810
                "Eriocheir sinensis reovirus": "no_mammals",
                # https://www.genome.jp/virushostdb/884200
                "Rotavirus D chicken/05V0049/DEU/2005": "avian",
                # https://www.genome.jp/virushostdb/59816
                "Yellowtail ascites virus": "no_mammals",
                # https://www.genome.jp/virushostdb/1002918
                "Simian sapelovirus 1": "non_primate_mammals",
                "simian sapelovirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/226613
                "Reptilian orthoreovirus": "no_mammals",
                # https://www.genome.jp/virushostdb/40067
                "Wad Medani virus": "no_mammals",
                # https://doi.org/10.1099/jgv.0.001837
                "Maraba virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1679172
                "Chobar Gorge virus": "no_mammals",
                # https://doi.org/10.3390/v14050905
                "Chirohepevirus eptesici": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2303488
                "Zegla virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1608075
                "Shuangao Insect Virus 1": "no_mammals",
                # https://doi.org/10.1128/jvi.06540-11
                "White-eye coronavirus HKU16": "avian",
                # https://www.genome.jp/virushostdb/10271
                "Rabbit fibroma virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/3052473
                "Orthohantavirus caobangense": "non_primate_mammals",
                # https://doi.org/10.1016/j.virol.2006.02.041
                "Pipistrellus bat coronavirus HKU5": "non_primate_mammals",
                # https://doi.org/10.3390/v14102258
                "Rice ragged stunt virus": "no_mammals",
                # https://doi.org/10.1128/genomea.00779-14
                "Loveridges garter snake virus 1": "no_mammals",
                # https://www.genome.jp/virushostdb/129950
                "Bovine mastadenovirus B": "non_primate_mammals",
                # https://doi.org/10.1007/BF01317540
                "Raccoonpox virus": "non_primate_mammals",
                # https://doi.org/10.1099/vir.0.058412-0
                "Cumuto virus": "no_mammals",
                # https://doi.org/10.1093/ilar.42.2.89
                "Woodchuck hepatitis virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2870351
                "limnipivirus A1": "no_mammals",
                # https://doi.org/10.1016/j.virusres.2009.09.013
                "Moussa virus": "no_mammals",
                # https://www.genome.jp/virushostdb/54315
                "Bovine viral diarrhea virus 2": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1406344
                "Kibale virus": "no_mammals",
                # https://doi.org/10.1016/B978-0-323-50934-3.00032-X
                "Feline leukemia virus": "non_primate_mammals",
                # https://doi.org/10.1111/avj.12659
                "Canid alphaherpesvirus 1": "non_primate_mammals",
                # https://doi.org/10.1128/jvi.02722-07
                "Beluga whale coronavirus SW1": "non_primate_mammals",
                # https://doi.org/10.1354/vp.38-6-644
                "Ovine adenovirus 7": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/666962
                "Drosophila obscura sigmavirus 10A": "no_mammals",
                # https://www.genome.jp/virushostdb/10991
                "Rice dwarf virus": "no_mammals",
                # https://doi.org/10.1038/s44298-024-00016-6
                "Orthohantavirus montanoense": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/465447
                "American grass carp reovirus": "no_mammals",
                # https://www.genome.jp/virushostdb/200402
                "Kamese virus": "no_mammals",
                # https://www.genome.jp/virushostdb/47680
                "Tree shrew adenovirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/329639
                "Canine minute virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1972566
                "Vesiculovirus radi": "no_mammals",
                # https://doi.org/10.1128/jvi.01121-10
                "Rousettus bat coronavirus HKU9": "non_primate_mammals",
                # https://doi.org/10.1128/jvi.01305-12
                "Rousettus bat coronavirus HKU10": "non_primate_mammals",
                # https://doi.org/10.7589/2017-03-067
                "Avian metaavulavirus 8": "avian",
                # https://doi.org/10.3389/fvets.2020.00419
                "Nairobi sheep disease virus": "non_primate_mammals",
                # https://doi.org/10.1099/jgv.0.001714
                "bovine rhinitis B virus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/11290
                "Infectious hematopoietic necrosis virus": "no_mammals",
                # https://doi.org/10.1002/hep.29494
                "Hepacivirus norvegici": "non_primate_mammals",
                # https://doi.org/10.1128/jvi.01142-12
                # aka bovine hungarovirus 1 
                "hunnivirus A1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/10366
                "Murid betaherpesvirus 1": "non_primate_mammals",
                # https://doi.org/10.1099/vir.0.051797-0
                "Turkey avisivirus": "avian",
                # https://www.genome.jp/virushostdb/1891770
                # related to Murine pneumotropic virus?
                # based on https://ictv.global/taxonomy/taxondetails?taxnode_id=20143115
                "Betapolyomavirus secumuris": "non_primate_mammals",
                # https://doi.org/10.1007/s11259-019-09746-y
                "Canine mastadenovirus A": "non_primate_mammals",
                # https://doi.org/10.1016/j.tim.2024.03.001
                # https://www.genome.jp/virushostdb/3052764
                # aka Rice hoja blanca tenuivirus
                "Tenuivirus oryzalbae": "no_mammals",
                # https://www.genome.jp/virushostdb/1987145
                "Shrew hepatovirus KS121232Sorara2012": "non_primate_mammals",
                # https://doi.org/10.3390/pathogens12121400
                "Hepacivirus vittatae": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/798074
                "Tuhoko virus 3": "non_primate_mammals",
                # https://doi.org/10.4269/ajtmh.14-0606
                "Keuraliba virus": "non_primate_mammals",
                # https://doi.org/10.1128/jvi.00371-09
                "Imjin virus": "non_primate_mammals",
                # https://doi.org/10.3390/v9050102
                "Mamastrovirus 13": "non_primate_mammals",
                # https://doi.org/10.1016/j.vetmic.2012.02.020
                "Marco virus": "no_mammals",
                # https://www.genome.jp/virushostdb/53182
                "Feline foamy virus": "non_primate_mammals",
                # https://doi.org/10.1128/mbio.01180-15
                "Phopivirus": "non_primate_mammals",
                # https://doi.org/10.1016/j.virusres.2017.09.002
                "Equid gammaherpesvirus 5": "non_primate_mammals",
                # https://doi.org/10.1016/j.cimid.2011.12.003 
                "Visna-maedi virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/248059
                "Abu Mina virus": "no_mammals",
                # https://doi.org/10.1111/j.1863-2378.2008.01208.x
                "Phnom Penh bat virus": "non_primate_mammals",
                # https://doi.org/10.1128/jvi.01094-13
                "Eel picornavirus 1": "no_mammals",
                # https://doi.org/10.1038/s41598-023-45205-0
                "Aquatic bird bornavirus 1": "avian",
                # https://doi.org/10.1371/journal.pone.0043881
                "Volepox virus": "non_primate_mammals",
                # https://doi.org/10.3390/pathogens12121400
                "Hepacivirus ratti": "non_primate_mammals",
                # https://doi.org/10.1099/vir.0.81584-0
                "Micromonas pusilla reovirus": "no_mammals",
                # https://doi.org/10.1016/j.vetmic.2015.07.024
                "Equine adenovirus 2": "non_primate_mammals",
                # https://doi.org/10.1007/s13337-013-0165-9
                "Bovine immunodeficiency virus": "non_primate_mammals",
                # https://doi.org/10.1038/sj.onc.1209850
                "Jaagsiekte sheep retrovirus": "non_primate_mammals",
                # https://doi.org/10.1007/s12250-018-0019-0
                # this article seems to suggest that the virus 
                # infected monkeys in a laboratory setting, 
                # but LLM query contradicts this assertion
                "Barur virus": "non_primate_mammals",
                # https://doi.org/10.1099/vir.0.066597-0
                "Chicken picornavirus 3": "avian",
                # https://www.genome.jp/virushostdb/62352
                "Eyach virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/660955
                "Nova virus": "non_primate_mammals",
                # https://doi.org/10.1007/s00705-007-1069-z
                "Main Drain virus": "non_primate_mammals",
                # https://doi.org/10.1016/j.onehlt.2021.100273
                "Mammarenavirus wenzhouense": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1611837
                "Chinook salmon bafinivirus": "no_mammals",
                # https://www.genome.jp/virushostdb/1664810
                "Ferak virus": "no_mammals",
                # https://doi.org/10.3390/v16030340
                "Mammarenavirus oliverosense": "non_primate_mammals",
                # https://doi.org/10.1016/j.virusres.2011.05.023
                "Murine adenovirus 2": "non_primate_mammals",
                # https://doi.org/10.1099/vir.0.053157-0
                "Sebokele virus 1": "non_primate_mammals",
                # https://doi.org/10.1128/jcm.02416-08
                "Cervid alphaherpesvirus 2": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/38804
                # https://doi.org/10.1128/jvi.77.24.13335-13347.2003
                "Yaba monkey tumor virus": "human",
                # https://doi.org/10.1016/j.meegid.2015.11.012
                "rabbit kobuvirus": "non_primate_mammals",
                # https://doi.org/10.1016/j.virol.2014.12.011
                # aka gairo virus
                "Mammarenavirus gairoense": "non_primate_mammals",
                # https://doi.org/10.1099/0022-1317-42-2-241
                "Drosophila X virus": "no_mammals",
                # https://doi.org/10.1128/jvi.01394-12
                "Miniopterus schreibersii picornavirus 1": "non_primate_mammals",
                # https://doi.org/10.1007/s00705-014-2113-4
                # found in fecal sample of bird known to eat mice
                # "host speacies not clear...most likely rodent born"
                "Mosavirus A2": "unknown",
                # https://www.genome.jp/virushostdb/10623
                # aka Shope papillomavirus (SPV)
                # aka cottontail rabbit papillomavirus (CRPV)
                "Kappapapillomavirus 2": "non_primate_mammals",
                # https://doi.org/10.1016/j.vetmic.2013.09.031
                "Equine infectious anemia virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/179241
                "Murine polyomavirus strain BG": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1308859
                "Drosophila affinis sigmavirus": "no_mammals",
                # https://www.genome.jp/virushostdb/1631554
                "Bovine nidovirus TCH5": "non_primate_mammals",
                # https://doi.org/10.1016/j.micpath.2023.106222
                "Respirovirus suis": "non_primate_mammals",
                # https://doi.org/10.1038/s41598-022-26103-3
                "Avian hepatitis E virus": "avian",
                # https://doi.org/10.1038/308291a0
                # aka Mastomys natalensis papillomavirus
                "Iotapapillomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/40537
                "Epsilonpapillomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/3052612
                "Pegivirus scotophili": "non_primate_mammals",
                # https://doi.org/10.1038/s41598-018-31193-z
                "Achimota virus 2": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1001080
                "Rockport virus": "non_primate_mammals",
                # https://doi.org/10.3201/eid0807.010281
                # aka bear valley virus
                "Mammarenavirus bearense": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/129956
                "Murine mastadenovirus A": "non_primate_mammals",
                # https://doi.org/10.1128/jvi.02364-10
                "Bat picornavirus BatPV/V1/13 Hun": "non_primate_mammals",
                # https://doi.org/10.1007/s00705-016-3188-x
                "Menghai rhabdovirus": "no_mammals",
                # https://doi.org/10.1007/s00705-019-04404-9
                "Penguin megrivirus": "avian",
                # https://doi.org/10.1007/s00705-017-3403-4
                "Harrier picornavirus 1": "avian",
                # https://doi.org/10.1128/mra.01207-20
                "Chicken megrivirus": "avian",
                # https://doi.org/10.1128/jvi.03342-14
                "Lesavirus 2": "primate",
                # https://doi.org/10.1128/jvi.03342-14
                "Lesavirus 1": "primate",
                # https://doi.org/10.3390/ijms25063272
                "Lymphocryptovirus Macaca/pfe-lcl-E3": "primate",
                # https://doi.org/10.1128/jvi.00651-16
                "Macaca nemestrina herpesvirus 7": "primate",
                # https://doi.org/10.1007/s00705-017-3696-3
                "Goose picornavirus 1": "avian",
                # https://doi.org/10.1128/jvi.79.16.10690-10700.2005
                "J-virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/310540
                # repeat entry of 'simian adenovirus 1'
                "Simian adenovirus 1": "human",
                # https://www.genome.jp/virushostdb/356862
                "Peruvian horse sickness virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/40012
                "Saumarez Reef virus": "no_mammals",
                # https://www.genome.jp/virushostdb/3052232
                "Hepacivirus myodae": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/798072
                "Tuhoko virus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/85708
                "Porcine circovirus 2": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1261100
                "Achimota virus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/710545
                "Durham virus": "avian",
                # https://www.genome.jp/virushostdb/3052594
                # aka avian paramyxovirus 3 
                "Paraavulavirus wisconsinense": "avian",
                # https://www.genome.jp/virushostdb/2656737
                "Lednice virus": "no_mammals",
                # https://www.genome.jp/virushostdb/871699
                "Passerivirus A1": "avian",
                # https://www.genome.jp/virushostdb/10406
                "Ground squirrel hepatitis virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1972570
                "Vesiculovirus malpais": "no_mammals",
                # https://www.genome.jp/virushostdb/10266
                "Sheeppox virus": "non_primate_mammals",
                # https://doi.org/10.3201/eid1912.121506
                "American bat vesiculovirus TFFN-2013": "non_primate_mammals",
                # repeat entry of aalivirus A1?
                # https://doi.org/10.3389/fvets.2024.1396552
                "Duck aalivirus 1": "avian",
                # https://doi.org/10.3389/fmicb.2019.00939
                "Rice yellow stunt nucleorhabdovirus": "no_mammals",
                # https://doi.org/10.1016/j.virusres.2009.06.003
                "Avian metaavulavirus 7": "avian",
                # repeat entry of 'callitrichine gammaherpesvirus 3'
                "Callitrichine gammaherpesvirus 3": "primate",
                # No mention of specific virus on VHDB
                # could refer to snakehead rhabdovirus
                # which is included already 
                "Snakehead virus": "unknown",
                # https://www.genome.jp/virushostdb/3052605
                "Pegivirus carolliae": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/490110
                "Curionopolis virus": "no_mammals",
                # https://www.genome.jp/virushostdb/10331
                "Equid alphaherpesvirus 4": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1972564
                "American bat vesiculovirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1633187
                "Laibin virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1552846
                "Alajuela virus": "no_mammals",
                # https://www.genome.jp/virushostdb/11673
                "Feline immunodeficiency virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/40053
                "Corriparta virus": "no_mammals",
                # https://www.genome.jp/virushostdb/12657
                "Equid gammaherpesvirus 2": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/70793
                "Turkey astrovirus": "avian",
                # https://www.genome.jp/virushostdb/490111
                # VHDB references cite isolation from Culicoides midges 
                # and subsequent cell culture study in mice
                "Itacaiunas virus": "no_mammals",
                # https://www.genome.jp/virushostdb/2847081
                "Fugong virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/144752
                "Mammarenavirus allpahuayoense": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1637496
                "Rotavirus I": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/80341
                "Equid alphaherpesvirus 3": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/40060
                "Umatilla virus": "avian",
                # https://www.genome.jp/virushostdb/10796
                "Porcine parvovirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1134579
                "Lunk virus NKS-1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/130499
                "Bovine atadenovirus D": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/11840
                "Gibbon ape leukemia virus": "primate",
                # https://www.genome.jp/virushostdb/990280
                "Jeju virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/3052235
                "Hepacivirus peromysci": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1482734
                "aalivirus A1": "avian",
                # https://www.genome.jp/virushostdb/1608144
                "Yichang Insect virus": "no_mammals",
                # https://www.genome.jp/virushostdb/766791
                "Mink coronavirus strain WD1127": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/11596
                "Hazara virus": "no_mammals",
                # https://www.genome.jp/virushostdb/1256869
                "Perhabdovirus perca": "no_mammals",
                # https://www.genome.jp/virushostdb/79700
                "rat cytomegalovirus strain Maastricht": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1144382
                # aka TtPV6
                "Tursiops truncatus papillomavirus 6": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/11970
                "Woolly monkey sarcoma virus": "primate",
                # https://www.genome.jp/virushostdb/59380
                # only infects rice plants
                "rice transitory yellowing virus": "no_mammals",
                # https://www.genome.jp/virushostdb/11977
                "Rabbit hemorrhagic disease virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/3052324
                # aka Pirital virus (PIRV)
                "Mammarenavirus piritalense": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1548718
                "Parrot bornavirus 4": "avian",
                # https://www.genome.jp/virushostdb/10381
                "Saimiriine gammaherpesvirus 2": "primate",
                # https://www.genome.jp/virushostdb/3052631
                "Phasivirus phasiense": "no_mammals",
                # https://www.genome.jp/virushostdb/11855
                "Mason-Pfizer monkey virus": "primate",
                # https://www.genome.jp/virushostdb/3052329
                "Mammarenavirus tamiamiense": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1987142
                "Rodent hepatovirus KEF121Sigmas2012": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/318558
                "Tupaia virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1978536
                "Ledantevirus nishimuro": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/10398
                "Ovine gammaherpesvirus 2": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1159904
                "Night heron coronavirus HKU19": "avian",
                # https://www.genome.jp/virushostdb/2560317
                "avian paramyxovirus 7": "avian",
                # https://www.genome.jp/virushostdb/35258
                "Lambdapapillomavirus 2": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1159902
                "Common moorhen coronavirus HKU21": "avian",
                # https://www.genome.jp/virushostdb/3052315
                "Mammarenavirus lunaense": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/380433
                "Kern Canyon virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/32615
                "Puma lentivirus 14": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1400425
                "Bowe virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2886894
                "Amapari virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/80939
                "Kairi virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/106331
                "callitrichine gammaherpesvirus 3": "primate",
                # https://doi.org/10.1016/j.virol.2003.09.021
                "Raza virus": "no_mammals",
                # https://www.genome.jp/virushostdb/201490
                # https://doi.org/10.1099/vir.0.009381-0
                # VHDB cites this paper as evidence of primate host, 
                # but there is no mention of Chlorocebus aethiops there
                "Equine encephalosis virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1987143
                "Bat hepatovirus BUO2BF86Colafr2010": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/47418
                "Phocid alphaherpesvirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/3052476
                "Orthohantavirus delgaditoense": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/64141
                "Porcine enterovirus 9": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/40051
                # https://www.science.org/content/article/livestock-virus-hits-europe-vengeance
                "Bluetongue virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/11757
                "Mouse mammary tumor virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/138184
                "Alcelaphine gammaherpesvirus 2": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/35265
                "Porcine adenovirus 3": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/3052729
                "Respirovirus bovis": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/64299
                "Jutiapa virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/35318
                "Patois virus": "no_mammals",
                # https://www.genome.jp/virushostdb/694007
                "Tylonycteris bat coronavirus HKU4": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/3052483
                "Orthohantavirus khabarovskense": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/64312
                "Montana myotis leukoencephalitis virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/11856
                "Squirrel monkey retrovirus": "primate",
                # https://www.genome.jp/virushostdb/10794
                "Minute virus of mice": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/35279
                # https://doi.org/10.1099/vir.0.014506-0
                # VHDB cites this paper as evidence of human host, but 
                # paper doesnt mention virus at all
                "Meaban virus": "avian",
                # https://www.genome.jp/virushostdb/104580
                "Kadipiro virus": "no_mammals",
                # https://www.genome.jp/virushostdb/1554501
                "Teviot virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1385427
                "Betacoronavirus Erinaceus/VMC/DEU/2012": "non_primate_mammals",
                # https://doi.org/10.7554/eLife.13135
                # https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?mode=Info&id=3052313&lvl=3&lin=f&keep=1&srchmode=1&unlock#note1
                # aka Loei River mammarenavirus
                "Mammarenavirus loeiense": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1620893
                "Iriri virus": "non_primate_mammals",
                # https://doi.org/10.1007/s13337-013-0145-0
                "Camelpox virus": "non_primate_mammals",
                # https://doi.org/10.3390/pathogens11101210
                "Porcine adenovirus 5": "non_primate_mammals",
                # https://doi.org/10.1016/j.bbrc.2004.09.154
                "Equus caballus papillomavirus 1": "non_primate_mammals",
                # https://doi.org/10.1111/j.1439-0426.1988.tb00562.x
                "Snakehead rhabdovirus": "no_mammals",
                # https://doi.org/10.3389/fcimb.2017.00212
                "avian paramyxovirus 4": "avian",
                # https://www.genome.jp/virushostdb/11318
                "Thogotovirus dhoriense": "human",
                # https://www.genome.jp/virushostdb/1303019
                "ROUT virus": "no_mammals",
                # https://www.genome.jp/virushostdb/1223562
                "Golden Gate virus": "no_mammals",
                # https://www.genome.jp/virushostdb/28274
                "Avian paramyxovirus 4": "avian",
                "avian paramyxovirus 4": "avian",
                # https://www.genome.jp/virushostdb/10788
                "Canine parvovirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1330520
                "Enterovirus F": "non_primate_mammals",
                # https://wwwn.cdc.gov/arbocat/VirusDetails.aspx?ID=400&SID=2
                "Rochambeau virus": "no_mammals",
                # https://www.genome.jp/virushostdb/3052319
                "Mammarenavirus merinoense": "non_primate_mammals",
                # https://doi.org/10.1128/mbio.01484-14
                "Ball python nidovirus 1": "no_mammals",
                # http://dx.doi.org/10.13005/bbra/1509
                "Jembrana disease virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2034827
                "cardiovirus C1": "non_primate_mammals",
                # https://doi.org/10.1016/j.cimid.2018.08.002
                "Mamastrovirus 5": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/188938
                "Pronghorn antelope pestivirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/3052311
                "Mammarenavirus latinum": "non_primate_mammals",
                # https://doi.org/10.1007/s11262-019-01694-8
                "Xipapillomavirus 1": "non_primate_mammals",
                # https://doi.org/10.1089/vbz.2013.1359
                "Anjozorobe virus": "non_primate_mammals",
                # https://doi.org/10.1099/vir.0.81258-0
                "Yunnan orbivirus": "no_mammals",
                # https://doi.org/10.3390/v14091973
                "Equine foamy virus": "non_primate_mammals",
                # https://doi.org/10.1099/vir.0.81090-0
                "Ectromelia virus": "non_primate_mammals",
                # https://doi.org/10.1099%2Fvir.0.047928-0
                "Eidolon helvum polyomavirus 1": "non_primate_mammals",
                # https://doi.org/10.1007/s00705-017-3565-0
                "Hedgehog dicipivirus": "non_primate_mammals",
                # https://doi.org/10.1128/genomea.00848-14
                "HCBI8.215 virus": "non_primate_mammals",
                # https://doi.org/10.1128/jvi.75.10.4854-4870.2001
                "Tupaiid betaherpesvirus 1": "non_primate_mammals",
                # https://doi.org/10.1080/03036751003641701
                "Whataroa virus": "avian",
                # https://doi.org/10.1016/j.virusres.2015.08.001
                "Bovine picornavirus": "non_primate_mammals",
                # https://doi.org/10.1016/j.hlife.2023.11.003
                "Bombali ebolavirus": "non_primate_mammals",
                # https://doi.org/10.3390/diseases9040073
                "Kaeng Khoi virus": "non_primate_mammals",
                # https://doi.org/10.1016/j.jviromet.2022.114638
                "Saboya virus": "no_mammals",
                # https://doi.org/10.1093/gbe/evab240
                "Drosophila immigrans sigmavirus": "no_mammals",
                # https://doi.org/10.1128%2FJVI.01858-14
                "Blacklegged tick phlebovirus 1": "no_mammals",
                # https://doi.org/10.1371/journal.pone.0118070
                "Bat circovirus POA/2012/II": "non_primate_mammals",
                # https://doi.org/10.1128%2FgenomeA.01393-15
                "Tadarida brasiliensis circovirus 1": "non_primate_mammals",
                # https://doi.org/10.1016/j.micpath.2023.106222
                "Orthorubulavirus suis": "non_primate_mammals",
                # https://doi.org/10.1371/journal.pone.0096934
                # Antibodies found in human subjects, "further work
                # needed to determine if these exposures result in virus
                # replication and/or clinical disease"
                "Turkey astrovirus 2": "avian",
                # https://doi.org/10.1128/jvi.01394-12
                "Rhinolophus ferrumequinum circovirus 1": "non_primate_mammals",
                # https://doi.org/10.3389/fmicb.2017.00786
                "Avian paramyxovirus UPO216": "avian",
                # https://doi.org/10.1371/journal.pone.0260360
                # "In our analyses, avian metaavulavirus 20 revealed the
                # highest degree of similarity with SARS-CoV-2 spike protein. "
                "Avian metaavulavirus 20": "avian",
                # https://doi.org/10.1016/j.virusres.2016.11.018
                "Avian paramyxovirus 14": "avian",
                # https://doi.org/10.1016/j.virol.2013.05.007
                "Mesocricetus auratus papillomavirus 1": "non_primate_mammals",
                # https://doi.org/10.7589/0090-3558-37.1.208
                "Anatid alphaherpesvirus 1": "avian",
                # https://doi.org/10.3390/v15091847
                "Feline infectious peritonitis virus": "non_primate_mammals",
                # https://doi.org/10.3390/v11090800
                "Ghana virus": "non_primate_mammals",
                # https://doi.org/10.1073/pnas.1516992112
                "Rodent hepatovirus RMU101637Micarv2010": "non_primate_mammals",
                # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5953381/
                "Gamboa virus": "avian",
                # https://doi.org/10.1186/s40168-017-0308-0
                # giant pandas
                "Aimelvirus 1": "non_primate_mammals",
                # https://doi.org/10.1111/tbed.12355
                "Avian metaavulavirus 5": "avian",
                # https://www.swinehealth.org/wp-content/uploads/2016/03/Porcine-sapelovirus-PSV.pdf
                "Porcine sapelovirus 1": "non_primate_mammals",
                # https://doi.org/10.7589/0090-3558-37.1.138
                "Aleutian mink disease virus": "non_primate_mammals",
                # https://ictv.global/report/chapter/pneumoviridae/pneumoviridae/orthopneumovirus
                "Bovine orthopneumovirus": "non_primate_mammals",
                # https://en.wikipedia.org/wiki/Royal_Farm_virus
                "Karshi virus": "human", 
                # https://www.merckvetmanual.com/poultry/viral-encephalitides-in-birds/israel-turkey-meningoencephalitis-virus-in-birds
                "Israel turkey meningoencephalomyelitis virus": "avian",
                # https://doi.org/10.1128%2FMRA.00551-21
                "Ndumu virus": "non_primate_mammals",
                # https://doi.org/10.1186/s13071-017-2559-9
                "Uganda S virus": "human",
                # https://doi.org/10.1186/s13071-017-2559-9
                # avian, but not known for humans:
                "Yaounde virus": "unknown",
                # https://doi.org/10.1186/s13071-018-2907-4
                # no known vertebrate host:
                "Aura virus": "unknown",
                # https://doi.org/10.4269/ajtmh.1980.29.1428
                "Fort Morgan virus": "avian",
                # https://wwwn.cdc.gov/arbocat/VirusDetails.aspx?ID=316&SID=7
                # Munguba antibodies may have been detected in a single
                # sloth (mammal) tested in Brazil years ago, but not enough info
                "Munguba virus": "unknown",
                "Tyuleniy virus": "unknown",
                # https://doi.org/10.4269%2Fajtmh.15-0065
                # and https://en.wikipedia.org/wiki/Entebbe_bat_virus
                # bats and mice, humans unknown
                "Entebbe bat virus": "non_primate_mammals",
                # https://doi.org/10.1016/j.meegid.2016.02.024
                # Capim found in opossums, but humans/primates
                # not clear
                "Capim virus": "non_primate_mammals",
                # I couldn't find conclusive evidence for this virus:
                "Orthobunyavirus simbuense": "unknown",
                "Equine adenovirus 1": "non_primate_mammals",
                "bovine respiratory syncytial virus": "non_primate_mammals",
                "Enterovirus J": "unknown",
                # https://doi.org/10.1371/journal.pntd.0010020
                "Middelburg virus":"human",
                # https://doi.org/10.1099/0022-1317-81-3-781
                "Apoi virus": "non_primate_mammals",
                "Southern elephant seal virus": "non_primate_mammals",
                # ICTV cites this isolate:
                # https://www.beiresources.org/Catalog/animalViruses/NR-10174.aspx
                # found in the rice rat, but not known to infect primates
                "Mammarenavirus cupixiense": "non_primate_mammals",
                # https://doi.org/10.1371/journal.pone.0004375
                "Gadgets Gully virus": "human",
                # https://en.wikipedia.org/wiki/Highlands_J_virus
                "Highlands J virus": "human",
                # https://en.wikipedia.org/wiki/Rio_Negro_virus#Host_interactions
                "Rio Negro virus": "human",
                # https://doi.org/10.1038/s41467-022-29298-1
                # borderline -- it can infect human cells in the lab
                # but no recorded natural infections I know of..
                "Cuevavirus lloviuense": "human",
                # https://doi.org/10.1016/j.virol.2007.09.022
                "Tellina virus 1": "no_mammals",
                "Jugra virus": "no_mammals",
                # https://doi.org/10.3390%2Fv4123494
                "Pneumonia virus of mice J3666": "non_primate_mammals",
                # antibodies have been detected against Morreton
                # in humans in Colombia
                # https://doi.org/10.1101/2022.03.10.483848
                "Vesiculovirus morreton": "human",
                "Joa virus": "unknown",
                # https://en.wikipedia.org/wiki/Macacine_gammaherpesvirus_4
                "Macacine gammaherpesvirus 4": "primate",
                # https://ictv.global/report_9th/RNAneg/Arenaviridae
                "Mopeia virus AN20410": "non_primate_mammals",
                # https://ictv.global/report_9th/RNAneg/Bunyaviridae
                # only mosquitoes for Bimiti confirmed so far it seems
                "Bimiti virus": "unknown",
                # https://en.wikipedia.org/wiki/Kedougou_virus
                "Kedougou virus": "human",
                "Adana virus": "unknown",
                # https://en.wikipedia.org/wiki/Cercopithecine_alphaherpesvirus_9
                "Cercopithecine alphaherpesvirus 9": "primate",
                # Cabassou means armadillo it seems; also known
                # as Venezuelan equine encephalitis virus V
                # based on:
                # https://ictv.global/report/chapter/togaviridae/togaviridae/alphavirus
                # so infects horses
                "Cabassou virus": "non_primate_mammals",
                # https://en.wikipedia.org/wiki/Kappapapillomavirus
                "Oryctolagus cuniculus papillomavirus 1": "non_primate_mammals",
                # https://en.wikipedia.org/wiki/Caprine_arthritis_encephalitis_virus
                # (goats and sheep only)
                "Caprine arthritis encephalitis virus": "non_primate_mammals",
                # https://doi.org/10.3390%2Fv13091730
                # fish only (so far):
                "Cutthroat trout virus": "no_mammals",
                # https://en.wikipedia.org/wiki/Yokose_virus#Host_and_location
                # only bats; requires manipulation to infect human cells
                "Yokose virus": "non_primate_mammals",
                # https://en.wikipedia.org/wiki/Infectious_pancreatic_necrosis_virus#Host_interactions
                "Infectious pancreatic necrosis virus": "no_mammals",
                # https://en.wikipedia.org/wiki/Pancreas_disease_in_farmed_salmon#Transmission
                "Salmon pancreas disease virus": "no_mammals",
                # https://en.wikipedia.org/wiki/Kadam_virus
                "Kadam virus": "non_primate_mammals",
                # https://en.wikipedia.org/wiki/Ungulate_bocaparvovirus_1
                "Bovine parvovirus": "non_primate_mammals",
                # https://doi.org/10.1099/0022-1317-71-9-2093
                # https://ictv.global/report/chapter/paramyxoviridae/paramyxoviridae/orthorubulavirus
                "Orthorubulavirus simiae": "human",
                # https://doi.org/10.7554%2FeLife.13135
                "Mammarenavirus ippyense": "non_primate_mammals",
                # https://en.wikipedia.org/wiki/Rotavirus#Virology
                "Rotavirus F chicken/03V0568/DEU/2003": "avian",
                # https://en.wikipedia.org/wiki/Enterovirus#Taxonomy indicates
                # Enterovirus H was formerly Simian enterovirus A, so probably
                # primate infectivity
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/enterovirus
                "Enterovirus H": "primate",
                # https://en.wikipedia.org/wiki/Tremovirus
                "tremovirus A1": "avian",
                # https://en.wikipedia.org/wiki/Akabane_orthobunyavirus
                "Akabane virus": "non_primate_mammals",
                # EHV has infected a human once, so let's make that a yes:
                # https://doi.org/10.4137%2Febo.s4966
                "Edge Hill virus": "human",
                # https://en.wikipedia.org/wiki/Aleutian_disease#
                # some controversy about rare human infection in:
                # https://doi.org/10.3201%2Feid1512.090514
                # but I wasn't convinced...
                "Aleutian mink disease parvovirus": "non_primate_mammals",
                # https://en.wikipedia.org/wiki/Tupavirus
                "Klamath virus": "human",
                "Mamastrovirus 3": "non_primate_mammals",
                # https://doi.org/10.3390%2Fv14020349
                "porcine sapelovirus 1": "non_primate_mammals",
                # https://en.wikipedia.org/wiki/Walleye_dermal_sarcoma_virus
                "Walleye dermal sarcoma virus": "no_mammals",
                # https://en.wikipedia.org/wiki/Rotavirus#Virology
                "Rotavirus G chicken/03V0567/DEU/2003": "avian",
                # https://ictv.global/report/chapter/flaviviridae/flaviviridae/pegivirus
                "Pegivirus sturnirae": "non_primate_mammals",
                # https://doi.org/10.3390/v9080203
                "Taterapox virus": "non_primate_mammals",
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/torchivirus
                # (tortoise host)
                "torchivirus A1": "no_mammals",
                # https://doi.org/10.3390%2Fv14092042
                "Epsilonpolyomavirus bovis": "non_primate_mammals",
                # https://www.ncbi.nlm.nih.gov/nuccore/KY013487.1
                # host is a primate
                "Mirim virus": "primate",
                # passage in mouse brain:
                # https://doi.org/10.1128%2FJVI.02080-08
                "Batama virus": "non_primate_mammals",
                # https://doi.org/10.1128/mbio.00180-12
                # is cited in the NCBI record (snakes)
                "CAS virus": "no_mammals",
                # https://en.wikipedia.org/wiki/Bovine_adenovirus
                "Bovine adenovirus 10": "non_primate_mammals",
                # https://doi.org/10.1089/vbz.2023.0029
                "Tacheng Tick Virus 1": "human",
                # fish: https://en.wikipedia.org/wiki/White_bream_virus
                "White bream virus": "no_mammals",
                # https://doi.org/10.3390%2Fv15030699
                "avian paramyxovirus 5": "avian",
                # https://wwwn.cdc.gov/arbocat/VirusDetails.aspx?ID=476&SID=5
                "Orthobunyavirus thimiriense": "avian",
                # https://en.wikipedia.org/wiki/Avian_metapneumovirus
                "Avian metapneumovirus": "avian",
                # https://en.wikipedia.org/wiki/Una_virus
                "Una virus": "human",
                # https://en.wikipedia.org/wiki/Phocine_morbillivirus
                "Phocine morbillivirus": "non_primate_mammals",
                # https://ictv.global/report/chapter/flaviviridae/flaviviridae/hepacivirus
                # (bats)
                "Hepacivirus macronycteridis": "non_primate_mammals",
                # https://ictv.global/report/chapter/polyomaviridae/polyomaviridae/alphapolyomavirus
                # (hamster)
                "Alphapolyomavirus mauratus": "non_primate_mammals",
                # https://doi.org/10.1002/jmv.21119
                # (voles)
                "Hantavirus Fusong-Mf-682": "non_primate_mammals",
                # https://doi.org/10.1099/0022-1317-77-12-3041
                # bovine (cattle plague)
                # only the second disease in history, after smallpox,
                # to be eradicated
                "Rinderpest virus (strain Kabete O)": "non_primate_mammals",
                # https://en.wikipedia.org/wiki/Panine_betaherpesvirus_2
                "Panine betaherpesvirus 2": "primate",
                # https://ictv.global/report/chapter/papillomaviridae/papillomaviridae/alphapapillomavirus
                "Alphapapillomavirus 12": "primate",
                # https://doi.org/10.1111%2Fzph.12459
                "Orthopoxvirus Abatino": "human",
                # https://ictv.global/report/chapter/polyomaviridae/polyomaviridae/alphapolyomavirus
                "Bat polyomavirus 5b": "non_primate_mammals",
                "Bat polyomavirus 6a": "non_primate_mammals",
                # https://doi.org/10.1099%2Fvir.0.029389-0
                # sandfly vector
                # can't find evidence of animal infection data
                "Aguacate virus": "no_mammals",
                # https://www.genome.jp/virushostdb/2016447
                # giant pandas?
                "ailurivirus A1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2016463
                "Giant panda polyomavirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/11582
                "Aino virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1725367
                # claims that there is a primate host, but the citation
                # doesn't seem to support it
                "Alcube virus": "no_mammals",
                # https://www.genome.jp/virushostdb/2847048
                # I actually can't find evidence that this virus
                # infects dogs, only the dog tick..
                # information is confounded by a glut of sites/resources
                # related to the tick itself
                "American dog tick virus": "no_mammals",
                # https://www.genome.jp/virushostdb/1642852
                # can't find evidence of animal host
                "Anadyr virus": "no_mammals",
                # https://www.genome.jp/virushostdb/104388
                # https://en.wikipedia.org/wiki/Duck_plague
                # this virus can propagate in Vero cells from a primate
                # but I don't think I've counted cell culture for
                # host range so far (should we?)
                # https://doi.org/10.1016/j.biologicals.2021.02.003
                "anatid alphaherpesvirus 1": "avian",
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/anativirus
                # https://www.genome.jp/virushostdb/2870379
                "anativirus B1": "avian",
                # https://en.wikipedia.org/wiki/Anhembi_orthobunyavirus
                # https://www.genome.jp/virushostdb/273355
                # The mouse host I buy, the primate seems to just be
                # a citation for Vero cell line, which lacks interferon
                "Anhembi virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/208491
                # https://ictv.global/report_9th/dsDNA/Polyomaviridae
                "Goose hemorrhagic polyomavirus": "avian",
                # https://doi.org/10.3390/v14071471
                # https://www.genome.jp/virushostdb/37325
                "Muscovy duck parvovirus": "avian",
                # https://www.genome.jp/virushostdb/50290
                # https://en.wikipedia.org/wiki/Aotine_betaherpesvirus_1
                "Aotine betaherpesvirus 1": "primate",
                # https://www.genome.jp/virushostdb/1606497
                # https://ictv.global/report/chapter/polyomaviridae/polyomaviridae/betapolyomavirus
                "Betapolyomavirus arplanirostris": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1606498
                # https://ictv.global/report/chapter/polyomaviridae/polyomaviridae/alphapolyomavirus
                "Alphapolyomavirus secarplanirostris": "non_primate_mammals",
                # https://ictv.global/report/chapter/polyomaviridae/taxonomy/polyomaviridae
                # https://www.genome.jp/virushostdb/1606499
                "Alphapolyomavirus tertarplanisrostris": "non_primate_mammals",
                # https://doi.org/10.1099%2Fvir.0.048850-0
                # propagated in Newborn mouse brain
                "Arumowot virus": "non_primate_mammals",
                # https://en.wikipedia.org/wiki/Asikkala_orthohantavirus
                "Orthohantavirus asikkalaense": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1236391
                "Alphapolyomavirus apaniscus": "primate",
                # https://www.genome.jp/virushostdb/133789
                # https://doi.org/10.3389/fvets.2021.813397
                "Budgerigar fledgling disease virus - 1": "avian",
                # https://en.wikipedia.org/wiki/Avian_coronavirus
                # https://www.genome.jp/virushostdb/11120
                "Infectious bronchitis virus Ind-TN92-03": "avian",
                # the evidence for primate here is cell culture
                # line: https://www.genome.jp/virushostdb/219704
                # not strong enough I don't think
                # https://doi.org/10.1128/jvi.77.12.6799-6810.2003
                "Avian adeno-associated virus ATCC VR-865": "avian",
                # this one also has primate label, but Vero cells
                # is the evidence (and they lack some interferon activity..)
                # https://www.genome.jp/virushostdb/1928005
                "avian paramyxovirus 14": "avian",
                # https://www.genome.jp/virushostdb/2560314
                "avian paramyxovirus 20": "avian",
                # https://www.genome.jp/virushostdb/11867
                # https://ictv.global/report/chapter/retroviridae/retroviridae/alpharetrovirus
                "Avian myelocytomatosis virus": "avian",
                # https://doi.org/10.1016%2Fj.heliyon.2019.e03099
                # https://www.genome.jp/virushostdb/2560322
                "avian paramyxovirus 16": "avian",
                # https://www.genome.jp/virushostdb/2094282
                # not much available about 17 it seems
                "Avian paramyxovirus 17": "avian",
                # https://www.genome.jp/virushostdb/11878
                # https://ictv.global/report/chapter/retroviridae/retroviridae/alpharetrovirus
                "Avian sarcoma virus CT10": "avian",
                # https://www.genome.jp/virushostdb/1634484
                # https://doi.org/10.1093/ve/vew037
                "Badger associated gemykibivirus 1": "non_primate_mammals",
                # https://ictv.global/report/chapter/rhabdoviridae/rhabdoviridae/barhavirus
                # https://doi.org/10.1099/0022-1317-67-6-1081
                "Bahia Grande virus": "human",
                # https://www.genome.jp/virushostdb/1972684
                # doesn't even grow in mammalian cell culture
                # https://doi.org/10.4269%2Fajtmh.16-0403
                "Almendravirus balsa": "no_mammals",
                # https://en.wikipedia.org/wiki/Hibecovirus
                # https://ictv.global/report/chapter/coronaviridae/coronaviridae/betacoronavirus
                # https://www.genome.jp/virushostdb/1541205
                "Bat Hp-betacoronavirus/Zhejiang2013": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1868218
                "Bat associated circovirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1329650
                "Bat circovirus": "non_primate_mammals",
                # https://ictv.global/report/chapter/circoviridae/circoviridae/circovirus
                # https://www.genome.jp/virushostdb/3052133
                "Circovirus siksparnis": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1868219
                "Bat associated circovirus 2": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1868220
                # https://ictv.global/report/chapter/circoviridae/circoviridae/circovirus
                "Bat associated circovirus 3": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2003309
                # https://ictv.global/report/chapter/circoviridae/circoviridae/circovirus
                "Bat associated circovirus 4": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2003310
                # https://ictv.global/report/chapter/circoviridae/circoviridae/circovirus
                "Bat associated circovirus 5": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2003311
                "Bat associated circovirus 6": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2003312
                # https://ictv.global/report/chapter/circoviridae/circoviridae/circovirus
                "Bat associated circovirus 7": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2003313
                "Bat associated circovirus 8": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2169823
                "Bat associated circovirus 9": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/795381
                "Bat cyclovirus GF-4c": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1572239
                "Bat circovirus POA/2012/VI": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1987736
                "Bat associated cyclovirus 11": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1987737
                "Bat associated cyclovirus 12": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1987738
                "Bat associated cyclovirus 13": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1795983
                "Pacific flying fox associated cyclovirus-1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1795984
                "Pacific flying fox associated cyclovirus-2": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1795985
                "Pacific flying fox associated cyclovirus-3": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2050585
                "Bat associated cyclovirus 2": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2050586
                "Bat associated cyclovirus 3": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2050587
                "Bat faeces associated cyclovirus 4": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/942035
                "Cyclovirus bat/USA/2009": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1987731
                "Bat associated cyclovirus 6": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1987732
                "Bat associated cyclovirus 7": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1987733
                "Bat associated cyclovirus 8": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1987734
                "Bat associated cyclovirus 9": "non_primate_mammals",
                # the human/primate evidence cited at
                # https://www.genome.jp/virushostdb/2758098
                # appears to be just cell culture lines...
                # https://en.wikipedia.org/wiki/Bat_mastadenovirus_A
                "bat adenovirus 3": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/696069
                # https://doi.org/10.1128%2FJVI.05974-11
                # bats, maybe canines
                "Bat adenovirus 2": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1788436
                # again, the human/primate evidence is cell
                # culture line only...
                "Bat mastadenovirus WIV9": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1788434
                "Bat mastadenovirus WIV12": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1788435
                "Bat mastadenovirus WIV13": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1986505
                "Bat mastadenovirus WIV17": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2015376
                "Bat mastadenovirus G": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2039267
                "Eidolon helvum adenovirus": "non_primate_mammals",
                # don't see much info on this one
                "Egyptian fruit bat adenovirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/740971
                # https://ictv.global/report/chapter/adenoviridae/adenoviridae/mastadenovirus
                "Bat mastadenovirus": "non_primate_mammals",
                # https://en.wikipedia.org/wiki/Batai_orthobunyavirus
                # https://doi.org/10.3390%2Fv14091868
                "Batai virus": "human",
                # https://www.genome.jp/virushostdb/889389
                "Calicivirus chicken/V0021/Bayern/2004": "avian",
                # https://en.wikipedia.org/wiki/Psittacine_beak_and_feather_disease
                "Beak and feather disease virus": "avian",
                # https://doi.org/10.1128%2FgenomeA.01485-16
                # not a natural infection of mouse though?
                "Beatrice Hill virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1922524
                "Beihai picobirna-like virus 7": "no_mammals",
                # evidence for human infection appears to be
                # from contaminated cell line:
                # https://doi.org/10.1016/j.virol.2007.01.045
                # rodent host seems more convincing
                "Beilong virus": "non_primate_mammals",
                # I don't usually count Vero cells as mammal/primate
                # evidence; the mosquito source is clear though
                "Bellavista virus": "no_mammals",
                # https://ictv.global/report/chapter/arteriviridae/arteriviridae/betaarterivirus
                # rodent evidence
                "RtEi arterivirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1806636
                # rodents
                "Rodent arterivirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/915424
                "Macaca fascicularis papillomavirus 2": "primate",
                # https://en.wikipedia.org/wiki/Bhanja_virus
                "Bhanja virus": "human",
                # https://www.genome.jp/virushostdb/273358
                # tricky though because broader species can
                # infect humans:
                # https://ictv.global/faqs
                # https://en.wikipedia.org/wiki/Bunyamwera_orthobunyavirus
                # but I don't see enough evidence for Biraro animal
                # infection yet...
                "Birao virus": "no_mammals",
                # https://www.genome.jp/virushostdb/1391037
                "Black robin associated gemykibivirus 1": "avian",
                # https://www.genome.jp/virushostdb/1985371
                "Blackbird associated gemycircularvirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1391038
                "Blackbird associated gemykibivirus 1": "avian",
                # https://www.genome.jp/virushostdb/1526521
                # can't find evidence for animal infection
                # ICTV says vertebate hosts not identified:
                # https://ictv.global/report/chapter/phenuiviridae/phenuiviridae/ixovirus
                "blacklegged tick virus 1": "no_mammals",
                # https://en.wikipedia.org/wiki/Bombali_ebolavirus
                # can infect human cells, but no compelling evidence
                # of actual human infections yet; natural host
                # seems to be bats
                "Bombali virus": "non_primate_mammals",
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/boosepivirus
                # cattle and sheep; limited information
                # https://doi.org/10.3390%2Fv16020307
                "boosepivirus A1": "non_primate_mammals",
                "boosepivirus B1": "non_primate_mammals",
                # https://doi.org/10.3201%2Feid2505.181573
                "Ovine picornavirus": "non_primate_mammals",
                # https://ictv.global/report/chapter/circoviridae/circoviridae/cyclovirus
                # https://www.genome.jp/virushostdb/942032
                "Cyclovirus PKbeef23/PAK/2009": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1843742
                "Faeces associated gemycircularvirus 22": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2004486
                "Bovine associated gemykibivirus 1": "non_primate_mammals",
                # https://doi.org/10.1093/ve/vew037
                # https://www.genome.jp/virushostdb/1516080
                "HCBI9.212 virus": "non_primate_mammals",
                # https://en.wikipedia.org/wiki/Bovine_gammaherpesvirus_6
                # https://www.genome.jp/virushostdb/1504288
                "Bovine gammaherpesvirus 6": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/273349
                # https://ictv.global/report/chapter/peribunyaviridae/taxonomy/peribunyaviridae
                "Bozo virus": "no_mammals",
                # https://www.genome.jp/virushostdb/667093
                # https://doi.org/10.1016/j.virol.2009.11.048
                "Broome reovirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/206377
                "Buenaventura virus": "no_mammals",
                # https://www.genome.jp/virushostdb/159140 suggests
                # insect only, but the primary literature:
                # https://doi.org/10.4269/ajtmh.1970.19.544
                # clearly suggests rabbits as hosts
                "Buttonwillow virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/629730
                # I don't see much useful info; search is polluted
                # by an unrelated plant virus
                "Cacao virus": "no_mammals",
                # https://www.genome.jp/virushostdb/80935
                # https://en.wikipedia.org/wiki/Cache_Valley_orthobunyavirus
                # https://doi.org/10.1128%2FJCM.00252-13
                "Cache Valley virus": "human",
                # https://www.genome.jp/virushostdb/1138490
                # https://ictv.global/report/chapter/peribunyaviridae/peribunyaviridae/orthobunyavirus
                "Cachoeira Porteira virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/3121209
                "cadicivirus B1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2572766
                "Caimito virus": "no_mammals",
                # https://www.genome.jp/virushostdb/142661
                "Canary circovirus": "avian",
                # https://en.wikipedia.org/wiki/Canarypox
                # https://www.genome.jp/virushostdb/44088
                # can enter but not replicate in human cells?
                "Canarypox virus": "avian",
                # https://www.genome.jp/virushostdb/1843735
                "Faeces associated gemycircularvirus 15": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1194757
                # https://en.wikipedia.org/wiki/Canine_circovirus
                # https://doi.org/10.1038/s41598-022-19815-z
                "Canine circovirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1980633
                "Betapolyomavirus canis": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1529392
                # https://doi.org/10.3389/fimmu.2020.01575
                "Caprine parainfluenza virus 3": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2163996
                # https://doi.org/10.1371/journal.pone.0199200
                "Capuchin monkey hepatitis B virus": "primate",
                # https://www.genome.jp/virushostdb/1891718
                "Alphapolyomavirus cardiodermae": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2870373
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/cardiovirus
                "cardiovirus F1": "non_primate_mammals",
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/parechovirus
                # Ljungan viruses appear to infect predominantly rodents (voles) and
                # have been proposed to infect humans, however, conclusive data is awaited. 
                # https://en.wikipedia.org/wiki/Parechovirus_B
                # primary literature discussion suggests possible connection to human disease, but
                # connection not strong enough yet and certainly not for this specific strain:
                # https://doi.org/10.1099%2Fvir.0.007948-0
                "Ljunganvirus 6": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1985417
                "Caribou associated gemykrogvirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1513315
                # https://ictv.global/report/chapter/parvoviridae/parvoviridae/amdoparvovirus
                # https://doi.org/10.3201%2Feid2012.140289
                "Raccoon dog amdovirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1908805
                "Amdoparvovirus sp.": "non_primate_mammals",
                # https://en.wikipedia.org/wiki/Carnivore_bocaparvovirus_1
                # https://www.genome.jp/virushostdb/1511885
                "Canine bocavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1174530
                "Feline bocavirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1882382
                "Sea otter parvovirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2758138
                "bat polyomavirus 4b": "non_primate_mammals",
                # https://doi.org/10.4103%2Fijmr.IJMR_1195_18
                # humans seem likely based on incidental reports
                # but not enough for me to be sure...
                # swine hosts in China seem confirmed
                "Cat Que virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1236392
                "Betapolyomavirus calbifrons": "primate",
                # https://www.genome.jp/virushostdb/1236395
                "Betapolyomavirus cercopitheci": "primate",
                # https://doi.org/10.1128/genomea.01081-15
                # https://www.genome.jp/virushostdb/1699094
                "Poecile atricapillus GI tract-associated gemycircularvirus": "avian",
                # https://www.genome.jp/virushostdb/12618
                # https://en.wikipedia.org/wiki/Chicken_anemia_virus
                "Chicken anemia virus": "avian",
                # https://www.genome.jp/virushostdb/942036
                "Cyclovirus NGchicken8/NGA/2009": "avian",
                # https://www.genome.jp/virushostdb/2109365
                "Chicken associated cyclovirus 2": "avian",
                # https://www.genome.jp/virushostdb/1843740
                "Faeces associated gemycircularvirus 20": "avian",
                # https://www.genome.jp/virushostdb/1843737
                "Faeces associated gemycircularvirus 17": "avian",
                # https://www.genome.jp/virushostdb/743290
                "Chimpanzee stool avian-like circovirus Chimp17": "primate",
                # https://www.genome.jp/virushostdb/742920
                "Cyclovirus Chimp11": "primate",
                # https://www.genome.jp/virushostdb/1590370
                "Betacoronavirus HKU24": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2849741
                # https://doi.org/10.1073/pnas.1908072116
                "Asian grey shrew hepatitis B virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/360397
                "Canis familiaris papillomavirus 3": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/464980
                "Canis familiaris papillomavirus 4": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1081055
                "Canis familiaris papillomavirus 8": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1131485
                # https://doi.org/10.1186/s12985-021-01677-y
                "Artibeus jamaicensis parvovirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1891719
                # https://ictv.global/report/chapter/polyomaviridae/polyomaviridae/alphapolyomavirus
                "Alphapolyomavirus chlopygerythrus": "primate",
                # https://www.genome.jp/virushostdb/1891758
                # https://ictv.global/report/chapter/polyomaviridae/polyomaviridae/betapolyomavirus
                "Betapolyomavirus secuchlopygerythrus": "primate",
                # https://www.genome.jp/virushostdb/1891720
                # https://ictv.global/report/chapter/polyomaviridae/polyomaviridae/alphapolyomavirus
                "Alphapolyomavirus tertichlopygerythrus": "primate",
                # https://www.genome.jp/virushostdb/2294094
                "Paguma larvata circovirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/93386
                # https://en.wikipedia.org/wiki/Columbid_alphaherpesvirus_1
                "Columbid alphaherpesvirus 1": "avian",
                # https://ictv.global/report/chapter/rhabdoviridae/rhabdoviridae/almendravirus
                # https://www.genome.jp/virushostdb/1972685
                "Almendravirus cootbay": "no_mammals",
                # https://www.genome.jp/virushostdb/349563
                "Crow polyomavirus": "avian",
                # https://ictv.global/report/chapter/poxviridae/poxviridae/oryzopoxvirus
                # https://www.genome.jp/virushostdb/930275
                "Cotia virus SPAn232": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1394033
                # https://ictv.global/report/chapter/polyomaviridae/polyomaviridae
                "Butcherbird polyomavirus": "avian",
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/crahelivirus
                "crahelivirus A1": "avian",
                # https://www.genome.jp/virushostdb/1764087
                # I don't count Vero cells though, interferon
                # compromised, etc.
                # potential human pathogen but I don't see concrete
                # evidence yet:
                # https://doi.org/10.1371/journal.pntd.0005978
                "Dashli virus": "unknown",
                # https://doi.org/10.1177%2F1040638718766036
                # https://www.genome.jp/virushostdb/78522
                "Odocoileus adenovirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2170000
                "Deer mastadenovirus B": "non_primate_mammals",
                # https://ictv.global/report/chapter/polyomaviridae/polyomaviridae/zetapolyomavirus
                # https://www.genome.jp/virushostdb/1891756
                "Zetapolyomavirus delphini": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/470214
                "Capreolus capreolus papillomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/996650
                "Camelus dromedarius papillomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1922325
                # https://doi.org/10.1016/j.vetmic.2016.12.035
                "Giraffa camelopardalis papillomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2758131
                "bat polyomavirus 2a": "non_primate_mammals",
                # https://doi.org/10.1038/srep12701
                # https://www.genome.jp/virushostdb/487311
                # https://doi.org/10.1186%2Fs13071-022-05341-4
                "Anopheles gambiae densovirus": "no_mammals",
                # https://ictv.global/report_9th/ssDNA/Parvoviridae
                # https://www.genome.jp/virushostdb/1513199
                # https://doi.org/10.1186/s40249-023-01099-8
                "Aedes albopictus densovirus 2": "no_mammals",
                # https://www.genome.jp/virushostdb/185638
                "Culex pipiens densovirus": "no_mammals",
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/diresapivirus
                "diresapivirus A1": "non_primate_mammals",
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/diresapivirus
                "diresapivirus B1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1623687
                "Bat polyomavirus 5a": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1623689
                "Bat polyomavirus 6b": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1623688
                "Bat polyomavirus 6c": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1714377
                "Bottlenose dolphin adenovirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2107574
                "Domestic cat hepadnavirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/663281
                "Drosophila melanogaster birnavirus SW-2009a": "no_mammals",
                # https://www.genome.jp/virushostdb/2006585
                "Duck associated cyclovirus 1": "avian",
                # https://en.wikipedia.org/wiki/Duck_atadenovirus_A
                # https://www.genome.jp/virushostdb/130328
                "Duck atadenovirus A": "avian",
                # https://www.genome.jp/virushostdb/1520006
                # https://en.wikipedia.org/wiki/Avian_adenovirus
                "Duck adenovirus 2": "avian",
                # https://en.wikipedia.org/wiki/Duck_circovirus
                # https://www.genome.jp/virushostdb/232802
                "Mulard duck circovirus": "avian",
                # https://www.genome.jp/virushostdb/300188
                "Duck coronavirus DK/GD/27/2014": "avian",
                # https://www.genome.jp/virushostdb/12639
                # https://en.wikipedia.org/wiki/Duck_hepatitis_B_virus
                "Duck hepatitis B virus": "avian",
                # https://www.genome.jp/virushostdb/1006585
                # I don't count Vero cells as primate infectino
                # because of interferon reduction, etc.
                "Durania virus": "no_mammals",
                # https://www.genome.jp/virushostdb/1163703
                "Equus asinus papillomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/446138
                "Sus scrofa papillomavirus 1": "non_primate_mammals",
                # https://ictv.global/report/chapter/papillomaviridae/papillomaviridae/dyoepsilonpapillomavirus
                # https://www.genome.jp/virushostdb/485362
                "Francolinus leucoscepus papillomavirus 1": "avian",
                # https://www.genome.jp/virushostdb/445217
                "Erinaceus europaeus papillomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/940834
                "Equus caballus papillomavirus 3": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1235428
                "Equus caballus papillomavirus 4": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/634772
                "Ovis aries papillomavirus 3": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1163708
                "Rupicapra rupicapra papillomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1887214
                "Bos taurus papillomavirus 16": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1887216
                "Bos taurus papillomavirus 18": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1747360
                "Pudu puda papillomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/759701
                "Bettongia penicillata papillomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/998829
                "Zalophus californianus papillomavirus 1": "non_primate_mammals",
                # https://doi.org/10.1093/gbe/evt211
                # https://www.genome.jp/virushostdb/1464072
                "Eptesicus serotinus papillomavirus 2": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/990304
                "Saimiri sciureus papillomavirus 1": "primate",
                # https://www.genome.jp/virushostdb/1338506
                "Talpa europaea papillomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/706527
                "Phocoena phocoena papillomavirus 4": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1464071
                "Eptesicus serotinus papillomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1235427
                "Equus caballus papillomavirus 6": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1352235
                "Castor canadensis papillomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1195364
                "Miniopterus schreibersii papillomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/568209
                "Felis domesticus papillomavirus 2": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1163701
                "Eidolon helvum papillomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1001533
                "Bos taurus papillomavirus 7": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1887217
                "Bos taurus papillomavirus 19": "non_primate_mammals",
                # https://doi.org/10.1038/s41598-017-16775-7
                # https://www.genome.jp/virushostdb/2042482
                "Eastern grey kangaroopox virus": "non_primate_mammals",
                # https://doi.org/10.7554/eLife.79777
                "Eidolon helvum bat coronavirus CMR704-P12": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1891722
                "Eidolon polyomavirus 1": "non_primate_mammals",
                # https://en.wikipedia.org/wiki/Elephant_endotheliotropic_herpesvirus
                # https://www.genome.jp/virushostdb/146015
                "Elephantid betaherpesvirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/548914
                "Elephant endotheliotropic herpesvirus 4": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/768738
                "Elephant endotheliotropic herpesvirus 5": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1552409
                "Sea otter polyomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/3052395
                "Orthobunyavirus guajaraense": "non_primate_mammals",
                # https://ictv.global/report/chapter/nairoviridae/nairoviridae/orthonairovirus
                # https://en.wikipedia.org/wiki/Ixodes_uriae
                "Tillamook virus": "no_mammals",
                # https://en.wikipedia.org/wiki/Squirrelpox_virus
                # https://www.genome.jp/virushostdb/240426
                "Squirrelpox virus": "non_primate_mammals",
                # https://doi.org/10.3390/pathogens9020116
                # https://www.genome.jp/virushostdb/154334
                "Macacine gammaherpesvirus 5": "primate",
                # https://ictv.global/report/chapter/hantaviridae/hantaviridae/mammantavirinae/mobatvirus
                # https://www.genome.jp/virushostdb/1841195
                "Quezon virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/11901
                # https://en.wikipedia.org/wiki/Bovine_leukemia_virus
                # https://doi.org/10.1016%2Fj.virusres.2023.199186
                # controversial--evidence of human infection...
                "Bovine leukemia virus": "human",
                # https://ictv.global/report/chapter/poxviridae/poxviridae/parapoxvirus
                # https://www.genome.jp/virushostdb/1579460
                "Parapoxvirus red deer/HL953": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/238817
                # I don't count Vero cells as sufficient evidence for
                # primate infection though
                # https://doi.org/10.3851%2FIMP1729
                # Maporal is intentionally used in hantavirus studies
                # to avoid BSL-4 requirements since there's no evidence
                # it can infect humans
                "Maporal virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/35314
                # mammal infection evidence not conclusive yet it seems:
                # https://doi.org/10.3390/v7112918
                "Koongol virus": "no_mammals",
                # https://www.genome.jp/virushostdb/50292
                # https://en.wikipedia.org/wiki/Cercopithecine_betaherpesvirus_5
                # viruses in this family do seem to be able to infect humans,
                # but I don't see concrete evidence of "strain 5" infecting
                # humans so I'm not confident enough...
                "Cercopithecine betaherpesvirus 5": "primate",
                # https://www.genome.jp/virushostdb/1821545
                # vertebrate hosts unknown as of 2022:
                # https://doi.org/10.3390%2Fv14050987
                "Enseada virus": "no_mammals",
                # https://doi.org/10.1038/srep28526
                # https://www.genome.jp/virushostdb/1826059
                "Enterovirus SEV-gx": "primate",
                # https://www.genome.jp/virushostdb/1434070
                # https://doi.org/10.1128/genomea.00298-18
                "Cervus papillomavirus 2": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1329402
                # (cell line not counted as primate evidence)
                # https://doi.org/10.1177/03009858241231556
                # no evidence of zoonotic activity:
                # https://ictv.global/report/chapter/poxviridae/poxviridae/vespertilionpoxvirus
                "Eptesipox virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1843738
                "Faeces associated gemycircularvirus 18": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1673646
                # https://ictv.global/report/chapter/picobirnaviridae/picobirnaviridae
                "Picobirnavirus Equ3": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1891761
                # https://ictv.global/report/chapter/polyomaviridae/polyomaviridae/betapolyomavirus
                "Betapolyomavirus equi": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2169903
                # this says "only birds:"
                # https://ictv.global/report/chapter/polyomaviridae/polyomaviridae/gammapolyomavirus
                "Erythrura gouldiae polyomavirus 1": "avian",
                # https://www.genome.jp/virushostdb/197771
                "Etapapillomavirus 1": "avian",
                # https://www.genome.jp/virushostdb/159143
                # don't see mammal evidence...
                "Facey's Paddock virus": "no_mammals",
                # https://www.genome.jp/virushostdb/685443
                # can't find evidence of anything other than eels infected...
                "Eel virus European X": "no_mammals",
                # https://www.genome.jp/virushostdb/1452540
                "Felis catus gammaherpesvirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1530454
                "Feline cyclovirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1108810
                "Feline picornavirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/400122
                "Finch circovirus": "avian",
                # https://www.genome.jp/virushostdb/2059380
                # poor mammalian cell culture growth and no evidence
                # of zoonotic activity:
                # https://ictv.global/report/chapter/poxviridae/poxviridae/avipoxvirus
                "Flamingopox virus FGPVKD09": "avian",
                # https://www.sciencedirect.com/topics/immunology-and-microbiology/aviadenovirus#:~:text=Aviadenoviruses%20are%20serologically%20distinct%20from,lengths%20in%20each%20penton%20base.
                # https://ictv.global/report/chapter/adenoviridae/adenoviridae/aviadenovirus
                # https://www.genome.jp/virushostdb/190061
                "Fowl aviadenovirus A": "avian",
                # https://www.genome.jp/virushostdb/172861
                "Fowl aviadenovirus 5": "avian",
                # https://www.genome.jp/virushostdb/190063
                "Fowl aviadenovirus C": "avian",
                # https://www.genome.jp/virushostdb/190064
                "Fowl aviadenovirus D": "avian",
                # https://www.genome.jp/virushostdb/172862
                # https://ictv.global/report/chapter/adenoviridae/adenoviridae/aviadenovirus
                "Fowl aviadenovirus 6": "avian",
                # https://www.genome.jp/virushostdb/10261
                # https://en.wikipedia.org/wiki/Fowlpox
                "Fowlpox virus": "avian",
                # https://www.genome.jp/virushostdb/11885
                # https://doi.org/10.1073%2Fpnas.77.4.2018
                "Fujinami sarcoma virus": "avian",
                # https://www.genome.jp/virushostdb/1985380
                # perhaps pigs as well:
                # https://doi.org/10.1007/s12250-020-00232-3
                "Fur seal associated gemycircularvirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/629736
                # https://doi.org/10.4269/ajtmh.1987.36.662
                "Gabek Forest virus": "non_primate_mammals",
                # https://en.wikipedia.org/wiki/Gallid_alphaherpesvirus_1
                # https://www.genome.jp/virushostdb/10386
                "Gallid alphaherpesvirus 1": "avian",
                # https://en.wikipedia.org/wiki/Marek%27s_disease
                # https://www.genome.jp/virushostdb/10390
                "Gallid alphaherpesvirus 2": "avian",
                # https://www.genome.jp/virushostdb/35250
                # https://ictv.global/report/chapter/orthoherpesviridae/orthoherpesviridae/mardivirus
                "Gallid alphaherpesvirus 3": "avian",
                # https://ictv.global/report/chapter/parvoviridae/parvoviridae/aveparvovirus
                # https://www.genome.jp/virushostdb/740934
                "Turkey parvovirus 260": "avian",
                # https://www.genome.jp/virushostdb/1846259
                # https://ictv.global/report/chapter/rhabdoviridae/rhabdoviridae/lyssavirus
                "Gannoruwa bat lyssavirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1985381
                "Gerygone associated gemycircularvirus 1": "avian",
                # https://www.genome.jp/virushostdb/1985382
                "Gerygone associated gemycircularvirus 2": "avian",
                # https://www.genome.jp/virushostdb/1985383
                "Gerygone associated gemycircularvirus 3": "avian",
                # https://www.genome.jp/virushostdb/942034
                # https://ictv.global/report/chapter/circoviridae/circoviridae/cyclovirus
                "Cyclovirus PKgoat11/PAK/2009": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1193422
                # https://ictv.global/report/chapter/adenoviridae/adenoviridae/aviadenovirus
                "Goose adenovirus 4": "avian",
                # https://www.genome.jp/virushostdb/146032
                "Goose circovirus": "avian",
                # https://www.genome.jp/virushostdb/2569586
                "Canada goose coronavirus": "avian",
                # https://www.genome.jp/virushostdb/928214
                # https://ictv.global/report/chapter/polyomaviridae/polyomaviridae
                "Gorilla gorilla gorilla polyomavirus 1": "primate",
                # https://www.genome.jp/virushostdb/487098
                # https://ictv.global/report/chapter/phenuiviridae/phenuiviridae/uukuvirus
                # looks like tick only?
                "Grand Arbaud virus": "no_mammals",
                # https://www.genome.jp/virushostdb/187984
                # https://en.wikipedia.org/wiki/Sealpox
                # https://doi.org/10.1111/j.1365-2133.2005.06451.x
                "Seal parapoxvirus": "human",
                # https://www.genome.jp/virushostdb/2870358
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/gruhelivirus
                "gruhelivirus A1": "avian",
                # https://www.genome.jp/virushostdb/2079601
                "Red-crowned crane parvovirus": "avian",
                # https://www.genome.jp/virushostdb/2870359
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/grusopivirus
                "grusopivirus A1": "avian",
                # https://www.genome.jp/virushostdb/2870360
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/grusopivirus
                "grusopivirus B1": "avian",
                # https://www.genome.jp/virushostdb/400121
                "Gull circovirus": "avian",
                # https://www.genome.jp/virushostdb/28300
                # https://doi.org/10.1128/jvi.62.10.3832-3839.1988
                # https://en.wikipedia.org/wiki/Avihepadnavirus
                "Heron hepatitis B virus": "avian",
                # I don't accept the cell culture line evidence
                # as strong enough, so counting as bats/mammals:
                # https://www.genome.jp/virushostdb/1554500
                "Hervey virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2021721
                # don't see evidence beyond mosquito
                "Holmes Jungle virus": "no_mammals",
                # https://www.genome.jp/virushostdb/1673638
                # https://ictv.global/report/chapter/circoviridae/circoviridae/cyclovirus
                "Cyclovirus Equ1": "non_primate_mammals",
                # may be a "naked virus" per
                # https://doi.org/10.1016/j.actatropica.2024.107158
                # https://www.genome.jp/virushostdb/1608048
                "Huangpi Tick Virus 2": "no_mammals",
                # https://www.genome.jp/virushostdb/310540
                # in this case I'm going to accept human based on the discussion
                # in:
                # https://doi.org/10.1128/jvi.01014-23
                # though this is a tough call...
                "simian adenovirus 1": "human",
                # https://www.genome.jp/virushostdb/273356
                "Iaco virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/159145
                # https://doi.org/10.1007/s13337-023-00808-z
                "Ingwavuma virus": "human",
                # https://www.genome.jp/virushostdb/1620892
                # neutralizing antibodies detected in rodents:
                # https://ictv.global/report/chapter/rhabdoviridae/rhabdoviridae/arurhavirus
                "Inhangapi virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1074206
                "Peromyscus papillomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1459044
                # I don't see evidence beyond tick hosts
                "Long Island tick rhabdovirus": "no_mammals",
                # https://www.genome.jp/virushostdb/655689
                "Itaituba virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/629735
                "Itaporanga virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1006586
                # I don't typically accept Vero cells as
                # solid evidence of primate infection
                # (prefer actual cases of infection)
                "Ixcanal virus": "no_mammals",
                # https://en.wikipedia.org/wiki/Jamestown_Canyon_encephalitis
                "Jamestown Canyon virus": "human",
                # https://doi.org/10.1016/S0168-1702(01)00262-3
                # https://www.genome.jp/virushostdb/150058
                "Jatobal virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/322067
                # https://ictv.global/report/chapter/paramyxoviridae/paramyxoviridae/jeilongvirus
                "J virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1851087
                # https://doi.org/10.1292%2Fjvms.21-0577
                "Kabuto mountain virus": "human",
                # https://www.genome.jp/virushostdb/35514
                # https://en.wikipedia.org/wiki/Keystone_virus
                # https://doi.org/10.4269/ajtmh.22-0594
                "Keystone virus": "human",
                # https://www.genome.jp/virushostdb/3052549
                # https://ictv.global/report/chapter/phasmaviridae/phasmaviridae/orthophasmavirus
                # appear to be insect-only
                "Orthophasmavirus kigluaikense": "no_mammals",
                # https://www.genome.jp/virushostdb/2847813
                # don't see decisive literature evidence for
                # mammal infection, though related viruses can
                # infect mammals
                "Kismaayo virus": "no_mammals",
                # https://www.genome.jp/virushostdb/2870349
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/kunsagivirus
                "kunsagivirus C1": "primate",
                # https://www.genome.jp/virushostdb/1985385
                "Lama associated gemycircularvirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1513269
                "Canis familiaris papillomavirus 6": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/312349
                "Procyon lotor papillomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1104917
                "Crocuta crocuta papillomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2027354
                # https://ictv.global/report/chapter/phenuiviridae/phenuiviridae/laulavirus
                "Laurel Lake virus": "no_mammals",
                # https://www.genome.jp/virushostdb/999729
                # cattle but not humans:
                # https://doi.org/10.1099%2Fvir.0.028308-0
                "Leanyer virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/481315
                # this viral family usually has a restricted host range:
                # (the virus tends to co-evolve with its host)
                # https://ictv.global/report/chapter/orthoherpesviridae/orthoherpesviridae
                "Leporid alphaherpesvirus 4": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1925019
                "Betapolyomavirus lepweddellii": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2847826
                # don't see much info on this one...
                "Leticia virus": "no_mammals",
                # https://www.genome.jp/virushostdb/1213198
                # https://doi.org/10.3201%2Feid1905.121071
                # not reported in humans.. yet..
                "Lleida bat lyssavirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2169904
                "Gammapolyomavirus lonmaja": "avian",
                # https://en.wikipedia.org/wiki/Lone_star_bandavirus
                # https://www.genome.jp/virushostdb/1219465
                # (I don't typically accept cell culture evidence)
                # I don't see concrete evidence of animal infection,
                # but I suspect it will show up eventually given the
                # tick was isolated feeding on a mammal...
                "Lone Star virus": "no_mammals",
                # https://www.genome.jp/virushostdb/2050037
                # no solid evidence for actual primate infection, but
                # cell lines susceptible:
                # https://doi.org/10.1073%2Fpnas.1308049110
                "Long-fingered bat hepatitis B virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2116559
                # https://ictv.global/report/chapter/paramyxoviridae/paramyxoviridae/jeilongvirus
                "Mount Mabu Lophuromys virus 1": "non_primate_mammals",
                "Mount Mabu Lophuromys virus 2": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1399914
                "African elephant polyomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1508224
                "Lucheng Rn rat coronavirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2079465
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/ludopivirus
                "ludopivirus A1": "avian",
                # https://www.genome.jp/virushostdb/80940
                # neutralizing antibodies in many humans
                # https://doi.org/10.1186/s13071-017-2559-9
                "Lumbo virus": "human",
                # https://www.genome.jp/virushostdb/1236398
                "Macaca fascicularis polyomavirus 1": "primate",
                # https://www.genome.jp/virushostdb/2560567
                "macacine betaherpesvirus 8": "primate",
                # https://www.genome.jp/virushostdb/2560568
                "macacine betaherpesvirus 9": "primate",
                # https://www.genome.jp/virushostdb/2560569
                "macacine gammaherpesvirus 10": "primate",
                # https://www.genome.jp/virushostdb/2560570
                # https://ictv.global/report/chapter/orthoherpesviridae/orthoherpesviridae/rhadinovirus
                "macacine gammaherpesvirus 11": "primate",
                # https://www.genome.jp/virushostdb/273352
                # https://doi.org/10.1099%2Fvir.0.039479-0
                "Macaua virus": "human",
                # https://www.genome.jp/virushostdb/11575
                # https://doi.org/10.3201%2Feid2308.161254
                "Maguari virus": "human",
                # https://www.genome.jp/virushostdb/2170064
                # https://ictv.global/report/chapter/spinareoviridae/spinareoviridae/orthoreovirus
                # I don't usually accept cell culture line evidence
                "Mahlapitsi orthoreovirus": "no_mammals",
                # https://www.genome.jp/virushostdb/1603963
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/malagasivirus
                "malagasivirus A1": "primate",
                # https://www.genome.jp/virushostdb/1603964
                "malagasivirus B1": "primate",
                # https://www.genome.jp/virushostdb/1985386
                "Mallard associated gemycircularvirus 1": "avian",
                # https://www.genome.jp/virushostdb/1843734
                "Faeces associated gemycircularvirus 14": "avian",
                # https://www.genome.jp/virushostdb/2495433
                # cell culture evidence not sufficient; bats
                # and ferret infection seems clear
                "Alston virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1654930
                # https://doi.org/10.1128%2FgenomeA.00781-15
                # CMVs are considered to be host restrictive due
                # to the lack of evidence of cross-species infection
                # and are thought to have co-evolved with their host.
                "Mandrillus leucophaeus cytomegalovirus": "primate",
                # https://www.genome.jp/virushostdb/3052318
                # https://ictv.global/report/chapter/arenaviridae/arenaviridae/mammarenavirus
                # could infect humans based on ICTV discussion, but don't
                # see explicit evidence for this particular virus
                "Mammarenavirus marientalense": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/391640
                # https://doi.org/10.1089%2Fvbz.2008.0131
                "Massilia virus": "no_mammals",
                # https://www.genome.jp/virushostdb/1891768
                "Betapolyomavirus mastomysis": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1775957
                # https://doi.org/10.1099/jgv.0.000389
                "Medjerda Valley virus": "human",
                # https://www.genome.jp/virushostdb/1477515
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/megrivirus
                "megrivirus C1": "avian",
                # https://www.genome.jp/virushostdb/2870375
                "megrivirus D1": "avian",
                # https://www.genome.jp/virushostdb/2079598
                "megrivirus E1": "avian",
                # https://www.genome.jp/virushostdb/35515
                # https://doi.org/10.1371%2Fjournal.pntd.0009494
                "Melao virus": "human",
                # https://www.genome.jp/virushostdb/37108
                "Meleagrid alphaherpesvirus 1": "avian",
                # https://www.genome.jp/virushostdb/1608323
                "Meles meles polyomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1919071
                # https://ictv.global/report/chapter/rhabdoviridae/rhabdoviridae/almendravirus
                "Menghai virus": "no_mammals",
                # https://www.genome.jp/virushostdb/159147
                # https://doi.org/10.4269/ajtmh.1981.30.473
                "Mermet virus": "avian",
                # https://www.genome.jp/virushostdb/1737523
                "Common vole polyomavirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1891769
                "Betapolyomavirus mafricanus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1985387
                "Miniopterus associated gemycircularvirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1904408
                "Miniopterus schreibersii polyomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1904409
                "Miniopterus schreibersii polyomavirus 2": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1475143
                "Mink circovirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2003500
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/mischivirus
                "mischivirus B1": "non_primate_mammals",
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/mischivirus
                "mischivirus D1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2758136
                "bat polyomavirus 3b": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1634487
                "Mongoose feces-associated gemycircularvirus d": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1634486
                "Mongoose feces-associated gemycircularvirus b": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1434720
                "Beluga whale alphaherpesvirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2161809
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/mosavirus
                "Marmot mosavirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2734558
                # https://doi.org/10.1016/j.virol.2021.01.007
                "Mosquito X virus": "no_mammals",
                # https://www.genome.jp/virushostdb/2304514
                "Culex circovirus-like virus": "no_mammals",
                # https://www.genome.jp/virushostdb/1034805
                "Mosquito VEM virus SDBVL G": "no_mammals",
                # https://www.genome.jp/virushostdb/241630
                # https://doi.org/10.1016/j.virol.2003.08.013
                # https://ictv.global/report/chapter/paramyxoviridae/paramyxoviridae
                # > Most viruses have a narrow host range in nature but can infect a broader range of cultured cells
                "Mossman virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2169856
                "Mouse associated cyclovirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2012619
                # not yet determined to pose a spillover risk as of 2024:
                # https://doi.org/10.1038/s41522-024-00543-3
                "Olivier's shrew virus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/932700
                # https://ictv.global/report/chapter/rhabdoviridae/rhabdoviridae/barhavirus
                # related viruses do infect humans, but I don't see hard evidence for MSV yet?
                # (just the original mosquito isolation)
                "Muir Springs virus": "no_mammals",
                # https://www.genome.jp/virushostdb/1569922
                # no live mammals infected in this study:
                # https://doi.org/10.3389%2Ffmicb.2022.791563
                # (just cell culture)
                # antibodies found in deer here:
                # https://doi.org/10.1016/j.ttbdis.2018.11.012
                "Mukawa virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/305674
                # https://ictv.global/report/chapter/poxviridae/poxviridae/cervidpoxvirus
                "Deerpox virus W-848-83": "non_primate_mammals",
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/mupivirus
                "mupivirus A1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1940555
                "Murine roseolovirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1261657
                # https://ictv.global/report/chapter/orthoherpesviridae/orthoherpesviridae/muromegalovirus
                "Murid betaherpesvirus 8": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/432370
                "Wood mouse herpesvirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/573199
                "Murine adenovirus 3": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2025359
                # https://ictv.global/report/chapter/poxviridae/poxviridae/centapoxvirus
                "Murmansk poxvirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1010668
                "Murre virus": "avian",
                # https://www.genome.jp/virushostdb/2171394
                "Mus musculus polyomavirus 3": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2050018
                "Rodent coronavirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1737522
                "Bank vole polyomavirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2116604
                # https://ictv.global/report/chapter/paramyxoviridae/paramyxoviridae/jeilongvirus
                "Pohorje myodes paramyxovirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2756244
                "bank vole virus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/563775
                "Myotis polyomavirus VM-2008": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1503289
                # https://ictv.global/report/chapter/coronaviridae/coronaviridae/alphacoronavirus
                "BtMr-AlphaCoV/SAX2011": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1920748
                "NL63-related bat coronavirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1172195
                # https://ictv.global/report/chapter/caliciviridae/caliciviridae/nacovirus
                "Turkey calicivirus": "avian",
                # https://www.genome.jp/virushostdb/590647
                # https://ictv.global/report/chapter/paramyxoviridae/paramyxoviridae/narmovirus
                "Nariva virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/38170
                # https://en.wikipedia.org/wiki/Avian_orthoreovirus
                # (ignore cell culture host range)
                "Avian orthoreovirus": "avian",
                # https://www.genome.jp/virushostdb/629739
                "Nique virus": "no_mammals",
                # https://www.genome.jp/virushostdb/2569589
                # https://doi.org/10.3201%2Feid2504.180750
                # https://doi.org/10.1155%2F2022%2F4231978
                "Ntepes virus": "human",
                # https://www.genome.jp/virushostdb/1503291
                "BtNv-AlphaCoV/SC2013": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/318852
                "Oak-Vale virus": "no_mammals",
                # https://www.genome.jp/virushostdb/1048855
                # https://ictv.global/report_9th/RNAneg/Bunyaviridae
                # can't find any support for animal infection of any kind,
                # just mosquito so far..
                "Odrenisrou virus": "no_mammals",
                # https://www.genome.jp/virushostdb/461322
                "Ursus maritimus papillomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1565990
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/orivirus
                "Chicken orivirus 1": "avian",
                # https://ictv.global/report_9th/RNAneg/Bunyaviridae
                # https://www.genome.jp/virushostdb/655691
                # I don't see evidence of infection beyond insects..
                "Oriximina virus": "no_mammals",
                # https://www.genome.jp/virushostdb/1391027
                "Faeces associated gemycircularvirus 12": "avian",
                # https://www.genome.jp/virushostdb/2035999
                "Otomops polyomavirus KY157": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2035998
                "Otomops polyomavirus KY156": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2601527
                "Ovine adenovirus 8": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1538454
                # https://ictv.global/report/chapter/peribunyaviridae/peribunyaviridae/pacuvirus
                "Pacui virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2035845
                "Chimpanzee polyomavirus Bob": "primate",
                # https://www.genome.jp/virushostdb/928211
                "Pan troglodytes verus polyomavirus 1a": "primate",
                # https://www.genome.jp/virushostdb/1891735
                "Alphapolyomavirus tertipanos": "primate",
                # https://www.genome.jp/virushostdb/1891736
                "Alphapolyomavirus quartipanos": "primate",
                # https://www.genome.jp/virushostdb/1891737
                "Alphapolyomavirus quintipanos": "primate",
                # https://www.genome.jp/virushostdb/1891738
                "Alphapolyomavirus sextipanos": "primate",
                # https://www.genome.jp/virushostdb/1891739
                "Alphapolyomavirus septipanos": "primate",
                # https://www.genome.jp/virushostdb/1762023
                "Pan troglodytes verus polyomavirus 8": "primate",
                # https://www.genome.jp/virushostdb/332937
                "Chimpanzee herpesvirus strain 105640": "primate",
                # https://www.genome.jp/virushostdb/1667587
                "Papio ursinus cytomegalovirus": "primate",
                # https://www.genome.jp/virushostdb/1286213
                "Yellow baboon polyomavirus 1": "primate",
                # https://www.genome.jp/virushostdb/1286214
                "Yellow baboon polyomavirus 2": "primate",
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/parabovirus
                "parabovirus A1": "non_primate_mammals",
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/parabovirus
                "parabovirus B1": "non_primate_mammals",
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/parabovirus
                "parabovirus C1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1964370
                "Falcon picornavirus": "avian",
                # https://www.genome.jp/virushostdb/1128118
                # https://ictv.global/report/chapter/hepadnaviridae/hepadnaviridae/avihepadnavirus
                "Parrot hepatitis B virus": "avian",
                # https://www.genome.jp/virushostdb/2065211
                # https://en.wikipedia.org/wiki/Passerivirus
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/passerivirus
                "Passerivirus sp.": "avian",
                # https://www.genome.jp/virushostdb/159151
                "Peaton virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1434088
                # https://ictv.global/report/chapter/adenoviridae/adenoviridae/siadenovirus
                "Chinstrap penguin adenovirus 2": "avian",
                # https://www.genome.jp/virushostdb/648998
                # https://en.wikipedia.org/wiki/Avipoxvirus
                # https://ictv.global/report/chapter/poxviridae/poxviridae/avipoxvirus
                "Penguinpox virus": "avian",
                # https://www.genome.jp/virushostdb/412969
                # https://doi.org/10.3390/v12060690
                "Porcine pestivirus isolate Bungowannah": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/119222
                # https://ictv.global/report/chapter/flaviviridaeport/flaviviridaeport/flaviviridae/pestivirus
                "Pestivirus giraffe-1 H138": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1562066
                "Norway rat pestivirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1689785
                "Porcine pestivirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/338903
                # https://doi.org/10.1016/j.virusres.2005.12.007
                # https://ictv.global/report/chapter/papillomaviridae/papillomaviridae/phipapillomavirus
                "Capra hircus papillomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/764030
                # https://ictv.global/report/chapter/adenoviridae/adenoviridae/aviadenovirus
                "Pigeon adenovirus 1": "avian",
                # https://www.genome.jp/virushostdb/1907767
                "Pigeon adenovirus 2": "avian",
                # https://www.genome.jp/virushostdb/126070
                "Columbid circovirus": "avian",
                # https://www.genome.jp/virushostdb/10264
                # https://ictv.global/report/chapter/poxviridae/poxviridae/avipoxvirus
                # https://en.wikipedia.org/wiki/Pigeon_pox
                "Pigeonpox virus": "avian",
                # https://www.genome.jp/virushostdb/1236406
                "Piliocolobus badius polyomavirus 2": "primate",
                # https://www.genome.jp/virushostdb/1236407
                "Piliocolobus rufomitratus polyomavirus 1": "primate",
                # https://www.genome.jp/virushostdb/1519097
                "Sesavirus CSL10538": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1427158
                "Seal parvovirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/392505
                "Mastomys coucha papillomavirus 2": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2850049
                "Apore virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/994672
                # https://doi.org/10.1371/journal.ppat.1002155
                "Titi monkey adenovirus ECC-2011": "human",
                # https://www.genome.jp/virushostdb/2848036
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/poecivirus
                # https://en.wikipedia.org/wiki/Avian_keratin_disorder
                "poecivirus A1": "avian",
                # https://www.genome.jp/virushostdb/2250215
                "Polar bear adenovirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2049933
                "Pomona bat hepatitis B virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1604875
                "Sumatran orang-utan polyomavirus": "primate",
                # https://www.genome.jp/virushostdb/1604874
                "Bornean orang-utan polyomavirus": "primate",
                # https://www.genome.jp/virushostdb/1843739
                "Faeces associated gemycircularvirus 19": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1391031
                "Faeces associated gemycircularvirus 2": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1868221
                # https://doi.org/10.1016/j.virusres.2022.198764
                "Porcine circovirus 3": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/273360
                # https://doi.org/10.4269/ajtmh.1998.59.704 (deer)
                # https://doi.org/10.7589/0090-3558-32.3.444 (deer)
                "Potosi virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/487097
                # scanning the literature, I'm only convinced
                # of infection of ticks... not animals (yet)
                "Precarious point virus": "no_mammals",
                # https://www.genome.jp/virushostdb/1581151
                "Slow loris parvovirus 1": "primate",
                # https://www.genome.jp/virushostdb/1219896
                # https://doi.org/10.3201/eid1901.121078
                "Raccoon polyomavirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1978920
                "Raccoon-associated polyomavirus 2": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/3052492
                # https://ictv.global/report/chapter/hantaviridae/hantaviridae/mammantavirinae/orthohantavirus
                # https://en.wikipedia.org/wiki/Prospect_Hill_orthohantavirus
                # https://doi.org/10.1016/0035-9203(87)90275-6
                # some serological evidence of infection in mammologists apparently...
                "Orthohantavirus prospectense": "human",
                # https://www.genome.jp/virushostdb/369584
                "Rousettus aegyptiacus papillomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1335476
                # https://ictv.global/report/chapter/papillomaviridae/papillomaviridae/psipapillomavirus
                "Eidolon helvum papillomavirus 2": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1335477
                # https://ictv.global/report/chapter/papillomaviridae/papillomaviridae/psipapillomavirus
                "Eidolon helvum papillomavirus 3": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/50294
                # https://ictv.global/report/chapter/herpesviridae/herpesviridae/iltovirus
                # https://en.wikipedia.org/wiki/Pacheco%27s_disease
                "Psittacid alphaherpesvirus 1": "avian",
                # https://www.genome.jp/virushostdb/1580497
                "Psittacine adenovirus 3": "avian",
                # https://www.genome.jp/virushostdb/2169709
                # https://ictv.global/report/chapter/adenoviridae/adenoviridae/aviadenovirus
                "Psittacine aviadenovirus B": "avian",
                # https://www.genome.jp/virushostdb/1891773
                "Betapolyomavirus ptedavyi": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2758132
                "bat polyomavirus 2b": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1343901
                # https://doi.org/10.1128/JVI.01277-14
                "Pteropodid alphaherpesvirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1873698
                # https://ictv.global/report/chapter/poxviridae/poxviridae/pteropopoxvirus
                # https://doi.org/10.1111/avj.13316
                "Pteropox virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1795989
                "Pacific flying fox faeces associated gemycircularvirus-10": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1985404
                "Pteropus associated gemycircularvirus 10": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1795994
                "Pacific flying fox faeces associated gemycircularvirus-2": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1985397
                "Pteropus associated gemycircularvirus 3": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1985398
                "Pteropus associated gemycircularvirus 4": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1985399
                "Pteropus associated gemycircularvirus 5": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1985400
                "Pteropus associated gemycircularvirus 6": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1985401
                "Pteropus associated gemycircularvirus 7": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1985402
                "Pteropus associated gemycircularvirus 8": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1985403
                "Pteropus associated gemycircularvirus 9": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1795988
                "Pacific flying fox faeces associated gemycircularvirus-1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1795991
                "Pacific flying fox faeces associated gemycircularvirus-12": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1795998
                "Pacific flying fox faeces associated gemycircularvirus-6": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1795999
                "Pacific flying fox faeces associated gemycircularvirus-7": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2758139
                "bat polyomavirus 5b1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1590650
                # https://ictv.global/report/chapter/polyomaviridae/polyomaviridae/gammapolyomavirus
                "Adelie penguin polyomavirus": "avian",
                # https://www.genome.jp/virushostdb/349564
                # https://ictv.global/report/chapter/polyomaviridae/polyomaviridae/gammapolyomavirus
                "Finch polyomavirus": "avian",
                # https://www.genome.jp/virushostdb/3052532
                # https://ictv.global/report/chapter/nairoviridae/nairoviridae/orthonairovirus
                # https://en.wikipedia.org/wiki/Qalyub_orthonairovirus
                # https://doi.org/10.4269/ajtmh.1985.34.180
                # looks like the ticks can at least transmit to mice
                # in the lab
                "Orthonairovirus qalyubense": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1391026
                "Faeces associated gemycircularvirus 11": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2682608
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/rabovirus
                "Rabovirus B1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2161803
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/sapelovirus
                "Marmot sapelovirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2682606
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/rabovirus
                "Rabovirus D1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1520002
                # https://doi.org/10.1016/j.meegid.2011.03.021
                # https://ictv.global/report/chapter/adenoviridae/adenoviridae/siadenovirus
                "Raptor adenovirus 1": "avian",
                # https://www.genome.jp/virushostdb/1708653
                "Gemycircularvirus gemy-ch-rat1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1679933
                "Rattus norvegicus polyomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1919247
                "Betapolyomavirus securanorvegicus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/345250
                # https://doi.org/10.1080/03079450600597345
                "Raven circovirus": "avian",
                # https://www.genome.jp/virushostdb/512169
                # https://en.wikipedia.org/wiki/Tulane_virus
                # https://ictv.global/report/chapter/caliciviridae/caliciviridae/recovirus
                # (I don't typically accept cell culture evidence)
                "Tulane virus": "primate",
                # https://www.genome.jp/virushostdb/2004965
                "Rhinolophus associated gemykibivirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2004966
                "Rhinolophus associated gemykibivirus 2": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1503292
                # https://ictv.global/report/chapter/coronaviridae/coronaviridae/alphacoronavirus
                "BtRf-AlphaCoV/HuB2013": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2908011
                "Trichechus manatus latirostris papillomavirus 1": "non_primate_mammals",
                # https://ictv.global/report/chapter/papillomaviridae/papillomaviridae/rhopapillomavirus
                # https://www.genome.jp/virushostdb/2848316
                "Trichechus manatus latirostris papillomavirus 3": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1972687
                # https://doi.org/10.4269/ajtmh.16-0403
                # https://ictv.global/report/chapter/rhabdoviridae/rhabdoviridae/almendravirus
                "Almendravirus chico": "no_mammals",
                # https://www.genome.jp/virushostdb/1538455
                "Rio Preto da Eva virus": "no_mammals",
                # https://www.genome.jp/virushostdb/3052131
                "Circovirus rongeur": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/3052126
                "Circovirus kiore": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/3052130
                "Circovirus roditore": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/3052132
                "Circovirus rosegador": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/3052122
                "Circovirus gnaver": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/3052120
                "Circovirus daga": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2305270
                "Bamboo rat circovirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2560714
                "Rodent associated cyclovirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2560715
                "Rodent associated cyclovirus 2": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1788315
                "Rat bocavirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2171381
                "Murine bocavirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2316143
                "Mouse kidney parvovirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2171377
                "Murine adeno-associated virus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2171378
                "Murine adeno-associated virus 2": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2847266
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/rohelivirus
                "rohelivirus A1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1902501
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/rosavirus
                "Rosavirus B": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1902502
                "Rosavirus C": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1929964
                "Rotavirus J": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1508710
                "Roundleaf bat hepatitis B virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/11886
                "Rous sarcoma virus": "avian",
                # https://www.genome.jp/virushostdb/1904411
                "Rousettus aegyptiacus polyomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1892416
                "Rousettus bat coronavirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/3052326
                # https://ictv.global/report/chapter/arenaviridae/arenaviridae/mammarenavirus
                # many mammarenaviruses have been reported to occasionally infect
                # exposed humans, but I don't see immediate evidence for
                # this one
                "Mammarenavirus ryukyuense": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/159138
                "Sabo virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/452475
                "Squirrel monkey polyomavirus": "primate",
                # https://www.genome.jp/virushostdb/1236410
                "Saimiri sciureus polyomavirus 1": "primate",
                # https://www.genome.jp/virushostdb/1535247
                # https://ictv.global/report/chapter/orthoherpesviridae/orthoherpesviridae/cytomegalovirus
                "Saimiriine betaherpesvirus 4": "primate",
                # https://www.genome.jp/virushostdb/2847279
                "Saint-Floris virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/520973
                # https://ictv.global/report/chapter/caliciviridae/caliciviridae/valovirus
                "Calicivirus pig/AB90/CAN": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/120499
                "Salem virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/427316
                "Salobo virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/45767
                "San Angelo virus": "no_mammals",
                # https://www.genome.jp/virushostdb/159152
                "Sango virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1620896
                # https://ictv.global/report/chapter/rhabdoviridae/rhabdoviridae/sawgrhavirus
                # can invade brains of experimentally infected mice
                "Sawgrass virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2848001
                "Nile warbler virus": "avian",
                # https://www.genome.jp/virushostdb/943083
                "California sea lion adenovirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1416741
                # https://ictv.global/report/chapter/poxviridae/poxviridae/mustelpoxvirus
                "Sea otter poxvirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/881945
                # https://ictv.global/report/chapter/polyomaviridae/polyomaviridae/gammapolyomavirus
                "Canary polyomavirus": "avian",
                # https://www.genome.jp/virushostdb/45768
                "Serra do Navio virus": "no_mammals",
                # https://www.genome.jp/virushostdb/1985407
                # isolated from sewage, no useful host info
                # that I can find...
                # label conservatively then
                "Sewage derived gemycircularvirus 1": "no_mammals",
                # https://www.genome.jp/virushostdb/1985408
                # similar analysis to case above
                "Sewage derived gemycircularvirus 2": "no_mammals",
                # https://www.genome.jp/virushostdb/1519401
                "Sewage-associated gemycircularvirus 6": "no_mammals",
                # https://www.genome.jp/virushostdb/1519405
                "Sewage-associated gemycircularvirus-7b": "no_mammals",
                # https://www.genome.jp/virushostdb/1519404
                "Sewage-associated gemycircularvirus 9": "no_mammals",
                # https://www.genome.jp/virushostdb/1519400
                # as above, remember that sewage could mean
                # replication in another organism that infects
                # animals, rather than direct infection...
                "Sewage-associated gemycircularvirus 5": "no_mammals",
                # https://www.genome.jp/virushostdb/1843761
                # the chicken host in the link above isn't
                # convincing to me, and most of the isolations
                # of these types of viruses so far could be
                # from gut microorganisms rather than the host proper
                "Sewage associated gemycircularvirus 3": "no_mammals",
                # https://www.genome.jp/virushostdb/1519407
                # again, no conclusive evidence for specific host
                # for this sewage-isolated sample
                "Sewage-associated gemycircularvirus 2": "no_mammals",
                # https://www.genome.jp/virushostdb/1519399
                "Sewage-associated gemycircularvirus 4": "no_mammals",
                # https://www.genome.jp/virushostdb/1843736
                "Faeces associated gemycircularvirus 16": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/291590
                # https://ictv.global/report/chapter/papillomaviridae/papillomaviridae/sigmapapillomavirus
                # https://doi.org/10.1016/j.virol.2004.10.033
                "Erethizon dorsatum papillomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1564099
                "Silverwater virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/38420
                # https://ictv.global/report/chapter/adenoviridae/adenoviridae/mastadenovirus
                "Simian adenovirus 3": "primate",
                # https://www.genome.jp/virushostdb/995022
                "Simian adenovirus 49": "primate",
                # https://www.genome.jp/virushostdb/1298385
                # https://doi.org/10.1128/mbio.00084-13
                "Baboon adenovirus 3": "human",
                # https://www.genome.jp/virushostdb/38432
                "Simian adenovirus 13": "primate",
                # https://www.genome.jp/virushostdb/1715778
                "Simian adenovirus 16": "primate",
                # https://www.genome.jp/virushostdb/909210
                "Simian adenovirus 18": "primate",
                # https://www.genome.jp/virushostdb/585059
                "Simian adenovirus 20": "primate",
                # https://www.genome.jp/virushostdb/1560346
                "Simian adenovirus DM-2014": "primate",
                # https://www.genome.jp/virushostdb/2848082
                "simian adenovirus 55": "primate",
                # https://www.genome.jp/virushostdb/1520004
                "Skua adenovirus 1": "avian",
                # https://www.genome.jp/virushostdb/1641642
                # https://doi.org/10.7589/JWD-D-21-00099
                # https://doi.org/10.1016/j.virol.2015.06.026
                "Skunk adenovirus PB1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/160796
                # I don't accept cell culture evidence for host range
                "Skunkpox virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/11580
                # https://doi.org/10.1093/jme/tjad128
                "Snowshoe hare virus": "human",
                # https://www.genome.jp/virushostdb/2050019
                # https://ictv.global/report/chapter/coronaviridae/coronaviridae/orthocoronavirinae
                "Shrew coronavirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2560769
                # https://en.wikipedia.org/wiki/Sorex_araneus_polyomavirus_1
                # https://doi.org/10.1016/j.isci.2021.103613
                # https://doi.org/10.1099/jgv.0.000948
                "Sorex araneus polyomavirus 1": "human",
                # https://www.genome.jp/virushostdb/2560770
                "Sorex coronatus polyomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2560771
                "Sorex minutus polyomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/273354
                "Sororoca virus": "no_mammals",
                # https://www.genome.jp/virushostdb/2010246
                # https://ictv.global/report/chapter/arenaviridae/arenaviridae/mammarenavirus
                "Souris virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2560777
                # https://ictv.global/report/chapter/orthoherpesviridae/orthoherpesviridae/mardivirus
                "Spheniscid alphaherpesvirus 1": "avian",
                # https://www.genome.jp/virushostdb/2170200
                # https://doi.org/10.1128/iai.8.5.804-813.1973
                # https://doi.org/10.1186/s12977-017-0379-9
                # https://doi.org/10.1371/journal.pone.0184502
                "Spider monkey simian foamy virus": "human",
                # https://www.genome.jp/virushostdb/1592764
                "Cyclovirus TsCyV-1_JP-NUBS-2014": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2773314
                "red squirrel adenovirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2847092
                # https://doi.org/10.3390/v10070373
                "giant squirrel virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1391025
                "Faeces associated gemycircularvirus 10": "avian",
                # https://www.genome.jp/virushostdb/349370
                "Starling circovirus": "avian",
                # https://www.genome.jp/virushostdb/1606500
                "Alphapolyomavirus sturnirae": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1491491
                # https://ictv.global/report/chapter/rhabdoviridae/rhabdoviridae/sunrhavirus
                "Sunguru virus": "avian",
                # https://www.genome.jp/virushostdb/459957
                "Swan circovirus": "avian",
                # https://www.genome.jp/virushostdb/45270
                # https://en.wikipedia.org/wiki/Tahyna_orthobunyavirus
                # https://doi.org/10.3201/eid1502.080722
                "Tahyna virus": "human",
                # https://www.genome.jp/virushostdb/2557875
                # https://doi.org/10.3390/v11030279
                # https://ictv.global/report/chapter/hepadnaviridae/hepadnaviridae/orthohepadnavirus
                "Tai Forest hepadnavirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1117633
                # https://ictv.global/report/chapter/paramyxoviridae/paramyxoviridae/jeilongvirus
                # https://doi.org/10.1128/JVI.06356-11
                "Tailam virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2026129
                # https://ictv.global/report/chapter/rhabdoviridae/rhabdoviridae/lyssavirus
                # https://doi.org/10.3390/v14071562
                "Taiwan bat lyssavirus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1538456
                # https://ictv.global/report/chapter/peribunyaviridae/peribunyaviridae/pacuvirus
                "Tapirape virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1623310
                # https://doi.org/10.1128/MRA.01248-18
                # https://doi.org/10.1016/0035-9203(81)90414-4
                "Tataguine virus": "human",
                # https://www.genome.jp/virushostdb/292792
                # https://ictv.global/report/chapter/papillomaviridae/papillomaviridae/taupapillomavirus
                # https://doi.org/10.1590/1678-4685-GMB-2021-0388
                # CPV infection is considered species-specific to dogs,
                # but oral papillomatosis has been described in two
                # members of the same Canidae family that the
                # subspecies Canis lupus familiaris belongs to,
                # namely the wolf and the coyote
                "Canis familiaris papillomavirus 2": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1226723
                "Canis familiaris papillomavirus 13": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1336600
                "Felis catus papillomavirus 3": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2259540
                "Mustela putorius papillomavirus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/273347
                # https://en.wikipedia.org/wiki/Tensaw_orthobunyavirus
                # https://doi.org/10.1007/s11262-009-0400-z
                "Tensaw virus": "human",
                # https://www.genome.jp/virushostdb/1508712
                # cell culture evidence for human, but I don't typically
                # accept that alone: https://doi.org/10.1073/pnas.1308049110
                "Tent-making bat hepatitis B virus": "non_primate_mammals",
                # https://ictv.global/report/chapter/picornaviridae/picornaviridae/teschovirus
                # Domestic pigs and wild boars are the only known hosts.
                "teschovirus B1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2773288
                # https://ictv.global/report/chapter/papillomaviridae/papillomaviridae/thetapapillomavirus
                "Psittacus erithacus papillomavirus 1": "avian",
                # https://www.genome.jp/virushostdb/1844928
                # looks like tick host info only
                "Avian-like circovirus": "unknown",
                # https://www.genome.jp/virushostdb/3052128
                # https://pubmed.ncbi.nlm.nih.gov/29567144/
                # looks like ticks only so far...
                "Circovirus pichong": "unknown",
                # https://www.genome.jp/virushostdb/1268011
                # https://ictv.global/report/chapter/hantaviridae/hantaviridae/mammantavirinae/orthohantavirus
                # https://doi.org/10.1089/vbz.2019.2452
                "Tigray virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1764085
                # https://doi.org/10.1371/journal.pntd.0004519
                # only sandfly seems to be confirmed as host
                "Toros virus": "unknown",
                # https://www.genome.jp/virushostdb/687385
                # https://ictv.global/report_9th/ssDNA/Anelloviridae
                "Torque teno canis virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/687383
                # https://ictv.global/report_9th/ssDNA/Anelloviridae
                "Torque teno douroucouli virus": "primate",
                # https://www.genome.jp/virushostdb/1673633
                "Torque teno equus virus 1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/687384
                # https://ictv.global/report_9th/ssDNA/Anelloviridae
                "Torque teno felis virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2065043
                "Torque teno felis virus 2": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2065056
                "Torque teno midi virus 15": "primate",
                # https://www.genome.jp/virushostdb/687372
                "Torque teno mini virus 4": "primate",
                # https://www.genome.jp/virushostdb/991022
                "Seal anellovirus TFFN/USA/2006": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1427157
                "Seal anellovirus 2": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1427156
                # https://en.wikipedia.org/wiki/Lambdatorquevirus
                "Seal anellovirus 3": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1566011
                "Seal anellovirus 4": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2012676
                # https://doi.org/10.1093/ve/vex017
                "Torque teno Leptonychotes weddellii virus-1": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/2012677
                "Torque teno Leptonychotes weddellii virus-2": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/687386
                # https://en.wikipedia.org/wiki/Iotatorquevirus
                # https://ictv.global/report_9th/ssDNA/Anelloviridae
                # https://en.wikipedia.org/wiki/Torque_teno_sus_virus
                # https://doi.org/10.3390/microorganisms10020242
                # zoonotic transmission pig -> ruminants, but I don't
                # see evidence for primate infection
                "Torque teno sus virus 1a": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/687387
                "Torque teno sus virus 1b": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1968861
                "Torque teno sus virus k2a": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/1239832
                "Torque teno sus virus k2b": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/687382
                # https://ictv.global/report_9th/ssDNA/Anelloviridae
                "Torque teno tamarin virus": "primate",
                # https://www.genome.jp/virushostdb/687381
                # https://ictv.global/report_9th/ssDNA/Anelloviridae
                "Torque teno tupaia virus": "non_primate_mammals",
                # https://www.genome.jp/virushostdb/687353
                "Torque teno virus 14": "primate",
                # https://www.genome.jp/virushostdb/687356
                "Torque teno virus 17": "human",
                # https://www.genome.jp/virushostdb/687357
                "Torque teno virus 18": "human",
                # https://www.genome.jp/virushostdb/687341
                "Torque teno virus 2": "primate",
                # https://www.genome.jp/virushostdb/687361
                # these "patent" TTV sequences are a bit suspicious?
                # how confident are we in human infection?
                # host-virus DB labels them as human though...
                "Torque teno virus 22": "human",
                # https://www.genome.jp/virushostdb/687362
                "Torque teno virus 23": "primate",
                # https://www.genome.jp/virushostdb/687364
                # above indicates human but the genbank
                # file indicates Japanese macaque
                "Torque teno virus 25": "primate",
                # https://www.genome.jp/virushostdb/687365
                "Torque teno virus 26": "primate",
                # TODO: verify -- was missing from Adam's data...
                # https://www.genome.jp/virushostdb/35279
                "Meaban virus": "avian",
                # TODO: verify -- was missing from Adam's data...
                # https://www.genome.jp/virushostdb/10794
                "Minute virus of mice": "non_primate_mammals",
                # TODO: verify -- was missing from Adam's data...
                "mischivirus A1": "non_primate_mammals",
                # TODO: verify -- was missing from Adam's data...
                "aichivirus E1": "non_primate_mammals",
                # TODO: verify -- was missing from Adam's data...
                "parechovirus C1": "non_primate_mammals",
                # TODO: verify -- was missing from Adam's data...
                "avisivirus C1": "avian",
                # TODO: verify -- was missing from Adam's data...
                "potamipivirus A1": "no_mammals",
                # TODO: verify -- was missing from Adam's data...
                "hepatovirus B1": "non_primate_mammals",
                # TODO: verify -- was missing from Adam's data...
                "avian paramyxovirus 8": "avian",
                # TODO: verify -- was missing from Adam's data...
                "Loveridge's garter snake virus 1": "no_mammals",
                # TODO: verify -- was missing from Adam's data...
                "Maize fine streak virus": "no_mammals",
                    # https://www.genome.jp/virushostdb/38143
    "Simian hemorrhagic fever virus": "primate",
    # https://www.genome.jp/virushostdb/666363
    "Drosophila melanogaster sigmavirus AP30": "no_mammals",
    # https://www.genome.jp/virushostdb/279896
    "Maize mosaic nucleorhabdovirus": "no_mammals",
    # https://www.genome.jp/virushostdb/200401
    # https://doi.org/10.1016/j.virol.2013.10.024 claims in Aves, can't find direct evidence
    "Hart Park virus": "no_mammals",
    # https://www.genome.jp/virushostdb/1936072
    "Kibale red colobus virus 2": "primate",
    # https://www.genome.jp/virushostdb/1620895
    # https://doi.org/10.1177/1176934317713484 source from above used for hamsters; only cell culture
    # https://www.ncbi.nlm.nih.gov/nuccore/NC_034543.1 isolated from mosquito
    "Ord River virus": "no_mammals",
        # https://www.genome.jp/virushostdb/3052234
    "Hepacivirus otomopis": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/336959
    "Chicken astrovirus": "avian",
    # https://www.genome.jp/virushostdb/28295
    # https://doi.org/10.1128/jcm.26.11.2235-2239.1988 only cell culture for primate
    "Porcine epidemic diarrhea virus": "non_primate_mammals",
    # https://www.ncbi.nlm.nih.gov/nuccore/NC_038543
    # https://doi.org/10.1371/journal.pone.0015113 only cell culture for primates
    "Chipmunk parvovirus": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/1972612
    # https://doi.org/10.1007/s00705-018-4003-7
    "Hapavirus flanders": "avian",
    # https://www.genome.jp/virushostdb/3052226
    "Hepacivirus bovis": "non_primate_mammals",
    # Alternate name Avian paramyxovirus 2
    # https://www.ncbi.nlm.nih.gov/nuccore/NC_039230.1/
    # https://doi.org/10.1016/j.virusres.2008.05.012 only cell culture for humans, etc
    "Avian metaavulavirus 2": "avian",
    "avian paramyxovirus 2": "avian",
    # https://www.genome.jp/virushostdb/1755590
    "Kunsagivirus A": "avian",
    # https://www.genome.jp/virushostdb/47001
    "Equine rhinitis B virus 1": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/1511784
    "Pasivirus A1": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/871700
    "Oscivirus A1": "avian",
    # https://www.genome.jp/virushostdb/3052238
    "Hepacivirus rhabdomysis": "non_primate_mammals",
    # https://doi.org/10.1099/vir.0.18731-0
    "Murid gammaherpesvirus 4": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/205895
    "Cypovirus 1": "no_mammals",
    # https://www.genome.jp/virushostdb/519497
    "Southern rice black-streaked dwarf virus": "no_mammals",
    # https://www.genome.jp/virushostdb/318834
    "Berrimah virus": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/1272946
    "La Joya virus": "no_mammals",
    # https://www.genome.jp/virushostdb/1307119
    "Theiler's disease-associated virus": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/1170234
    "Feline morbillivirus": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/11048
    "Lactate dehydrogenase-elevating virus": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/11636
    "Reticuloendotheliosis virus": "avian",
    # https://www.genome.jp/virushostdb/10995
    "Infectious bursal disease virus": "avian",
    # https://www.genome.jp/virushostdb/2885846
    "Alces alces papillomavirus 1": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/1965066
    "Porcine reproductive and respiratory syndrome virus 1": "non_primate_mammals",
    "porcine reproductive and respiratory syndrome virus 1": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/1157337
    "Piscine orthoreovirus": "no_mammals",
    # https://www.genome.jp/virushostdb/1965067
    "Porcine reproductive and respiratory syndrome virus 2": "non_primate_mammals",
    # Alternate name Aquareovirus ctenopharyngodontis
    # https://www.ncbi.nlm.nih.gov/nuccore/NC_005166.1/
    # https://www.genome.jp/virushostdb/185782
    "Aquareovirus C": "no_mammals",
    "Aquareovirus ctenopharyngodontis": "no_mammals",
    # https://www.genome.jp/virushostdb/1416021
    "Feline sakobuvirus A": "non_primate_mammals",
    # Alternate name maize Iranian mosaic virus
    # https://www.ncbi.nlm.nih.gov/nuccore/NC_036390.1
    # https://www.genome.jp/virushostdb/348823
    "Maize Iranian mosaic nucleorhabdovirus": "no_mammals",
    "maize Iranian mosaic virus": "no_mammals",
    # https://www.genome.jp/virushostdb/1272958
    # https://doi.org/10.3389/fmicb.2019.00856 antibodies in water buffalo
    "Sweetwater Branch virus": "no_mammals",
        # https://www.genome.jp/virushostdb/994485
    "Scophthalmus maximus reovirus": "no_mammals",
    # https://www.genome.jp/virushostdb/3052762
    "Tenuivirus oryzabrevis": "no_mammals",
    # https://www.genome.jp/virushostdb/1605403
    "Non-primate hepacivirus NZP1": "non_primate_mammals",
    # https://www.ncbi.nlm.nih.gov/nuccore/NC_038292
    "Kibale red-tailed guenon virus 1": "primate",
    # https://www.genome.jp/virushostdb/380160
    # https://doi.org/10.1016/j.virusres.2022.198739 in cattle, PCR
    "Obodhiang virus": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/10353
    "Saimiriine alphaherpesvirus 1": "primate",
      # https://www.genome.jp/virushostdb/471285
    "Lettuce yellow mottle virus": "no_mammals",
    # https://www.genome.jp/virushostdb/2847088
    "GB virus-D": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/40061
    # https://doi.org/10.1111/j.1751-0813.1999.tb12126.x also seems to be in kangaroos, other marsupials
    "Wallal virus": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/1479613
    "Puerto Almendras virus": "no_mammals",
    # https://doi.org/10.1007/s11262-012-0819-5
    # isolated from sandflies, it's assumed to infect mammals due to the family, but no direct evidence
    "Vesiculovirus bogdanovac": "no_mammals",
    # https://www.genome.jp/virushostdb/311228
    "Mycoreovirus 1": "no_mammals",
    # https://www.genome.jp/virushostdb/32612
    "Lettuce necrotic yellows virus": "no_mammals",
    # https://www.genome.jp/virushostdb/2682610
    "Zambian malbrouck virus 1": "primate",
    # https://doi.org/10.1099/vir.0.058172-0
    "Carp picornavirus 1": "no_mammals",
    # https://doi.org/10.1371/journal.pone.0097730
    "Enterovirus E": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/31612
    "Adelaide River virus": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/318836
    # https://doi.org/10.1016/j.virol.2012.01.004 probably in cattle, serological evidence only
    "Kotonkan virus": "no_mammals",
        # https://www.genome.jp/virushostdb/694001
    "Miniopterus bat coronavirus HKU8": "non_primate_mammals",
    # https://doi.org/10.1038/293543a0
    "Moloney murine leukemia virus": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/1641896
    "Falcovirus A1": "avian",
    # https://www.genome.jp/virushostdb/1231903
    # https://doi.org/10.1128/jvi.01856-14 its limited host range is known/notable
    "Eilat virus": "no_mammals",
    # https://www.genome.jp/virushostdb/1272943
    "Joinjakaka virus": "no_mammals",
    # https://www.genome.jp/virushostdb/1330070
    "Melegrivirus A": "avian",
    # https://www.genome.jp/virushostdb/1354498
    "Guereza hepacivirus": "primate",
    # https://www.genome.jp/virushostdb/185954
    "Mal de Rio Cuarto virus": "no_mammals",
    # https://www.genome.jp/virushostdb/1158189
    "Chaco virus": "no_mammals",
    # https://www.genome.jp/virushostdb/11978
    "Feline calicivirus": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/75888
    "Baboon orthoreovirus": "primate",
    # https://doi.org/10.1007/s00705-005-0652-4
    "Tenuivirus persotritici": "no_mammals",
        # https://www.genome.jp/virushostdb/1755588
    "Sicinivirus A": "avian",
    # https://doi.org/10.1371/journal.pone.0108379
    "Warrego virus": "non_primate_mammals",
    # https://doi.org/10.1128/jvi.74.9.4264-4272.2000
    "Koala retrovirus": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/764599
    "Coastal Plains virus": "non_primate_mammals",
    # Alternate name avisivirus B1
    # https://www.genome.jp/virushostdb/1534545
    "Chicken picornavirus 2": "avian",
    "avisivirus B1": "avian",
    # https://www.genome.jp/virushostdb/1659220
    "Kobuvirus cattle/Kagoshima-1-22-KoV/2014/JPN": "non_primate_mammals",
    # detected in pool of insects, insect vector assumed to transmit to bats, but no further evidence
    # https://doi.org/10.7554/eLife.05378
    "Wuhan Louse Fly Virus 5": "no_mammals",
    # https://doi.org/10.1016/j.virusres.2005.07.005
    "Breda virus": "non_primate_mammals",
    # https://doi.org/10.1099/vir.0.83605-0
    "Miniopterus bat coronavirus 1": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/3052236
    # difficult to parse; but it seems tamarins were infected with a human GB hepatitis agent to sequence
    # https://doi.org/10.1073/pnas.92.8.3401
    # reference for the GB agent used
    # https://doi.org/10.1093/infdis/142.5.767
    "Hepacivirus platyrrhini": "human",
    # https://www.genome.jp/virushostdb/1348439
    "Niakha virus": "no_mammals",
    # https://www.genome.jp/virushostdb/1384461
    "Bat coronavirus CDPHE15/USA/2006": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/1159905
    "Porcine coronavirus HKU15": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/318843
    # there is evidence for humans, but only positive neutralization test
    # https://doi.org/10.1016/j.virep.2014.09.001
    "Almpiwar virus": "no_mammals",
    # https://www.genome.jp/virushostdb/129953
    "Bovine mastadenovirus A": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/1987139
    "Bat hepatovirus SMG18520Minmav2014": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/133704
    "Porcine circovirus 1": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/2870363
    "anativirus A1": "avian",
    # https://www.genome.jp/virushostdb/77698
    "Fiji disease virus": "no_mammals",
    # https://www.genome.jp/virushostdb/202910
    "Bubaline alphaherpesvirus 1": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/358764
    "Border disease virus": "non_primate_mammals",
       # https://www.genome.jp/virushostdb/1884879
    "Parrot bornavirus 5": "avian",
    # https://www.genome.jp/virushostdb/1620897
    "Sripur virus": "no_mammals",
    # https://www.genome.jp/virushostdb/2847078
    "Felis domesticus papillomavirus 1": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/3052229
    "Hepacivirus glareoli": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/1965064
    "African pouched rat arterivirus": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/205908
    "Cypovirus 14": "no_mammals",
    # https://www.genome.jp/virushostdb/572288
    "Bulbul coronavirus HKU11-934": "avian",
    # https://doi.org/10.1007/s00705-013-1735-2
    "Avian orthoavulavirus 12": "avian",
        # Alternate name avian paramyxovirus 10
    # https://www.genome.jp/virushostdb/862945 primate evidence is cell culture only
    "Avian paramyxovirus penguin/Falkland Islands/324/2007": "avian",
    "avian paramyxovirus 10": "avian",
    # https://doi.org/10.1371/journal.pone.0087593
    "Fathead minnow picornavirus": "no_mammals",
    # https://www.genome.jp/virushostdb/10989
    "Maize rough dwarf virus": "no_mammals",
    # https://www.genome.jp/virushostdb/1664809
    "Jonchet virus": "no_mammals",
    # https://www.genome.jp/virushostdb/1406343
    "Tai virus": "no_mammals",
    # https://www.genome.jp/virushostdb/1550518
    "Koolpinyah virus": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/402584
    "Bovine viral diarrhea virus 3 Th/04_KhonKaen": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/572289
    "Munia coronavirus HKU13-3514": "avian",
    # https://doi.org/10.1186/s12985-017-0839-9
    "Green River chinook virus": "no_mammals",
        # https://www.genome.jp/virushostdb/1734441
    "Rat arterivirus 1": "non_primate_mammals",
    # VHDB has different accession for this name; maybe just different isolates
    # Source of this accession:
    # https://doi.org/10.1007/BF00284644
    "Tenuivirus zeae": "no_mammals",
    # https://www.genome.jp/virushostdb/205899
    "Cypovirus 5": "no_mammals",
    # https://www.genome.jp/virushostdb/3085316
    "Bos taurus papillomavirus 1": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/1150861
    "Canine picodicistrovirus": "non_primate_mammals",
    # https://doi.org/10.1186/1743-422X-10-228
    # other isolates have been shown to infect humans in vitro:
    # https://doi.org/10.1007/s11262-009-0377-7
    "Porcine endogenous retrovirus": "non_primate_mammals",
    # https://doi.org/10.1099/vir.0.055939-0
    "Kolente virus": "non_primate_mammals",
    # https://doi.org/10.1016/j.vetmic.2017.04.024
    "Deltapapillomavirus 3": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/1546177
    "Mikumi yellow baboon virus 1": "primate",
    # https://www.genome.jp/virushostdb/11307
    "Sonchus yellow net nucleorhabdovirus": "no_mammals",
    # https://www.genome.jp/virushostdb/55987
    "Isavirus salaris": "no_mammals",
        # https://www.genome.jp/virushostdb/1221391
    "Cedar virus": "non_primate_mammals",
    # https://doi.org/10.1128/jvi.75.3.1186-1194.2001
    "Bovine gammaherpesvirus 4": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/1272947
    "Landjia virus": "avian",
    # https://www.genome.jp/virushostdb/696863
    "Sprivivirus cyprinus": "no_mammals",
    # https://www.genome.jp/virushostdb/10986
    "Rice gall dwarf virus": "no_mammals",
    # https://www.genome.jp/virushostdb/381543
    "Salmon aquaparamyxovirus": "no_mammals",
    # https://www.genome.jp/virushostdb/162013
    # Serological evidence in humans:
    # https://doi.org/10.1086/520817
    "Tioman virus": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/1608146
    "Yongjia Tick Virus 2": "no_mammals",
    # https://www.genome.jp/virushostdb/3052609
    "Pegivirus neotomae": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/1272957
    "Sena Madureira virus": "no_mammals",
    # https://www.genome.jp/virushostdb/173082
    "Chum salmon reovirus CS": "no_mammals",
    # https://www.genome.jp/virushostdb/79891
    "Cervid alphaherpesvirus 1": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/10990
    "Rice black streaked dwarf virus": "no_mammals",
    # https://www.genome.jp/virushostdb/118140
    "Teschovirus A": "non_primate_mammals",
       # https://www.genome.jp/virushostdb/1118369
    "Wobbly possum disease virus": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/1195087
    "Sunshine Coast virus": "no_mammals",
    # https://www.genome.jp/virushostdb/39637
    "Equid alphaherpesvirus 8": "non_primate_mammals",
    # https://doi.org/10.1128/jvi.65.6.2910-2920.1991
    "Equine arteritis virus": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/104581
    "St Croix River virus": "no_mammals",
    # https://www.genome.jp/virushostdb/11096
    "Classical swine fever virus": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/1537975
    # Serological evidence for humans
    # https://doi.org/10.1128/jvi.02932-14
    "Kumasi rhabdovirus": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/3052627
    "Phasivirus baduense": "no_mammals",
    # https://www.genome.jp/virushostdb/1987144
    "Hedgehog hepatovirus Igel8Erieur2014": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/195060
    "Potato yellow dwarf nucleorhabdovirus": "no_mammals",
    # https://www.genome.jp/virushostdb/318849
    "Fukuoka virus": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/10273
    "Myxoma virus": "non_primate_mammals",
    # https://doi.org/10.1038/ncomms1796
    "Bat Paramyxovirus Epo_spe/AR1/DRC/2009": "non_primate_mammals",
    # https://doi.org/10.1371/journal.ppat.1004664
    "Le Dantec virus": "human",
        # https://www.genome.jp/virushostdb/311229
    "Mycoreovirus 3": "no_mammals",
    # https://www.genome.jp/virushostdb/137443
    "Macropodid alphaherpesvirus 1": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/1775456
    "Colocasia bobone disease-associated virus": "no_mammals",
    # https://doi.org/10.1371/journal.ppat.1004664
    "Carajas virus": "no_mammals",
    # https://doi.org/10.3390/diseases11010021
    "Yata virus": "no_mammals",
    # https://www.genome.jp/virushostdb/207343
    "Bovine foamy virus": "non_primate_mammals",
    # https://doi.org/10.1016/j.meegid.2015.11.012
    "Rabbit picornavirus": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/380434
    "Mount Elgon bat virus": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/1474807
    # closest phylogenetic relative, Langya henipavirus, infects humans:
    # https://doi.org/10.1056/nejmc2202705
    "Mojiang virus": "non_primate_mammals",
    # https://doi.org/10.1016/s0378-1135(96)01231-x
    "Bovine alphaherpesvirus 1": "non_primate_mammals",
       # https://www.genome.jp/virushostdb/693999
    "Scotophilus bat coronavirus 512": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/194965
    # cell culture only for humans
    "Aichivirus B": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/1173138
    "Influenza D virus (D/swine/Oklahoma/1334/2011)": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/40065
    # some serological evidence for 'non_primate_mammals':
    # https://doi.org/10.3390/v7052185
    "Chenuda virus": "no_mammals",
    # https://www.genome.jp/virushostdb/3052559
    "Orthorubulavirus mapueraense": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/2905917
    # claim of human host not corroborated elsewhere
    "Pichinde virus": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/1972683
    "Almendravirus arboretum": "no_mammals",
       # https://www.genome.jp/virushostdb/10334
    "Felid alphaherpesvirus 1": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/2908018
    "Taro vein chlorosis virus": "no_mammals",
    # https://www.genome.jp/virushostdb/35243
    "Ateline alphaherpesvirus 1": "primate",
    # https://www.genome.jp/virushostdb/200404
    "Mossuril virus": "no_mammals",
    # https://www.genome.jp/virushostdb/195059
    "Datura yellow vein nucleorhabdovirus": "no_mammals",
    # Alternate name Fikirini virus
    # https://www.genome.jp/virushostdb/1408144
    "Fikirini rhabdovirus": "non_primate_mammals",
    "Fikirini virus": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/3052496
    "Orthohantavirus sangassouense": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/1972569
    "Vesiculovirus perinet": "no_mammals",
        # There is serological evidence in humans
    # Additionally, a volunteer infected with symptoms:
    # https://doi.org/10.1007/BF01246390
    "Equine rhinitis A virus": "human",
    # Alternate name avian paramyxovirus 9
    # non-avian evidence cell culture only
    # https://www.genome.jp/virushostdb/2560326
    "Avian orthoavulavirus 9": "avian",
    "avian paramyxovirus 9": "avian",
    # https://www.genome.jp/virushostdb/3052448
    "Orthobunyavirus teteense": "avian",
    # UniProt evidence for human can't be corroborated
    # https://www.genome.jp/virushostdb/340907
    "Papiine alphaherpesvirus 2": "primate",
    # Alternate naem avian paramyxovirus 13
    # non-avian evidence cell culture only
    # https://www.genome.jp/virushostdb/2560321
    "Avian paramyxovirus goose/Shimane/67/2000": "avian",
    "avian paramyxovirus 13": "avian",
    # https://www.genome.jp/virushostdb/1987141
    "Rodent hepatovirus CIV459Lopsik2004": "non_primate_mammals",
      # https://www.genome.jp/virushostdb/1159908
    "Wigeon coronavirus HKU20": "avian",
    # https://www.genome.jp/virushostdb/889873
    "Fathead minnow nidovirus": "no_mammals",
    # https://www.genome.jp/virushostdb/1985704
    "Cytorhabdovirus gramineae": "no_mammals",
    # https://www.genome.jp/virushostdb/1511807
    # https://doi.org/10.1016/j.virol.2014.01.018
    "Rosavirus A2": "human",
    "rosavirus A2": "human",
    # https://www.genome.jp/virushostdb/380442
    # Paper claims isolated from human:
    # https://doi.org/10.1016/S0769-2617(87)80041-2
    "Nkolbisson virus": "human",
    # https://www.genome.jp/virushostdb/1985699
    "Cytorhabdovirus hordei": "no_mammals",
    # isolated from wallabies
    # https://doi.org/10.1371/journal.pone.0031911
    "Eubenangee virus": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/1679445
    "Bruges virus": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/1715288
    "Canary bornavirus 1": "avian",
    # https://www.genome.jp/virushostdb/55744
    "Equid alphaherpesvirus 9": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/569195
    "Porcine kobuvirus swine/S-1-HUN/2007/Hungary": "non_primate_mammals",
        # https://www.genome.jp/virushostdb/134606
    "Trichoplusia ni cypovirus 15": "no_mammals",
    # https://www.genome.jp/virushostdb/3052731
    "Respirovirus muris": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/152177
    "Dichorhavirus orchidaceae": "no_mammals",
    # https://www.genome.jp/virushostdb/1272942
    "Gray Lodge virus": "no_mammals",
    # https://www.genome.jp/virushostdb/237020
    "Porcine torovirus": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/390157
    "Senecavirus A": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/449133
    "Epizootic hemorrhagic disease virus (serotype 1 / strain New Jersey)": "non_primate_mammals",
    # evidence for macropods, but only serological survey
    # https://www.genome.jp/virushostdb/1972623
    "Hapavirus ngaingan": "no_mammals",
    # evidence for mammals, but only cell study
    # https://www.genome.jp/virushostdb/246280
    "Liao ning virus": "no_mammals",
    # https://www.genome.jp/virushostdb/11303
    "Bovine ephemeral fever virus": "non_primate_mammals",
      # https://www.genome.jp/virushostdb/1985706
    "Cytorhabdovirus fragariarugosus": "no_mammals",
    # https://www.genome.jp/virushostdb/40050
    "African horse sickness virus": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/1382295
    "Inachis io cypovirus 2": "no_mammals",
    # Note: not RefSeq, and there appears to be RefSeq available (NC_001408)
    # https://doi.org/10.3382/ps/pey231
    "Avian leukosis virus": "avian",
    # https://doi.org/10.3390/pathogens10091143
    "Dabieshan virus": "no_mammals",
    # https://www.genome.jp/virushostdb/1661395
    "Ampivirus A1": "no_mammals",
    # https://www.genome.jp/virushostdb/2907837
    "Fer-de-lance virus": "no_mammals",
    # https://www.genome.jp/virushostdb/1288358
    "Tench rhabdovirus": "no_mammals",
    # https://www.genome.jp/virushostdb/33706
    "Caviid betaherpesvirus 2": "non_primate_mammals",
        # https://www.genome.jp/virushostdb/2764087
    "gallivirus A1": "avian",
    # Alternate name parechovirus D or parechovirus D1
    # https://www.genome.jp/virushostdb/1394142
    "Ferret parechovirus": "non_primate_mammals",
    "parechovirus D1": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/1239574
    # https://doi.org/10.1099/vir.0.19267-0
    "Mamastrovirus 10": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/47929
    # mention of human cell evidence:
    # https://doi.org/10.1002/ajpa.24604
    "Macacine betaherpesvirus 3": "primate",
    "macacine betaherpesvirus 3": "primate",
    # https://www.genome.jp/virushostdb/33756
    # https://doi.org/10.1099/0022-1317-77-8-1693
    "European brown hare syndrome virus": "non_primate_mammals",
    # Alternate name avian paramyxovirus 6
    # https://www.genome.jp/virushostdb/2560316
    "Avian metaavulavirus 6": "avian",
    "avian paramyxovirus 6": "avian",
    # https://www.genome.jp/virushostdb/1980915
    # https://doi.org/10.1016/j.virusres.2004.06.004
    "Novirhabdovirus hirame": "no_mammals",
    # https://www.genome.jp/virushostdb/3052325
    # Alternate name Mobala virus:
    # https://doi.org/10.1371/journal.pone.0072290
    "Mammarenavirus praomyidis": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/35244
    "Bovine alphaherpesvirus 5": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/1518603
    "Arenaviridae sp. 13ZR68": "non_primate_mammals",
    # https://doi.org/10.1128/jvi.00589-18
    "Berne virus": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/114429
    "Bovine adenovirus 2": "non_primate_mammals",
       # https://www.genome.jp/virushostdb/2593991
    "Peste des petits ruminants virus": "non_primate_mammals",
    "peste-des-petits-ruminants virus": "non_primate_mammals",
    # Alternate names: pseudorabies virus, Aujeszky’s disease virus
    # https://www.genome.jp/virushostdb/10345
    # Infection of humans is controversial:
    # https://doi.org/10.3390/pathogens9080633
    "Suid alphaherpesvirus 1": "non_primate_mammals",
    # Alternate name: Eggplant mottled dwarf virus
    # https://www.genome.jp/virushostdb/488317
    "Eggplant mottled dwarf nucleorhabdovirus": "no_mammals",
    "Eggplant mottled dwarf virus": "no_mammals",
    # https://www.genome.jp/virushostdb/1538452
    "Feline astrovirus D1": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/3052763
    "Tenuivirus oryzaclavatae": "no_mammals",
    # Alternate name: Dinovernavirus aedis
    # https://www.genome.jp/virushostdb/341721
    # https://doi.org/10.1016/j.virol.2005.08.028
    "Aedes pseudoscutellaris reovirus": "no_mammals",
    "Dinovernavirus aedis": "no_mammals",
    # https://www.genome.jp/virushostdb/35252
    "Alcelaphine gammaherpesvirus 1": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/798073
    "Tuhoko virus 2": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/37131
    "Dolphin morbillivirus": "non_primate_mammals",
    # https://doi.org/10.1023/a:1008140707132
    "Viral hemorrhagic septicemia virus Fil3": "no_mammals",
    # https://www.genome.jp/virushostdb/2756242
    "Aydin-like pestivirus": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/1006061
    "Duck hepatitis A virus 1": "avian",
      "Equine pegivirus 1": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/318845
    "Parry Creek virus": "no_mammals",
    # https://www.genome.jp/virushostdb/1972626
    "Hapavirus wongabel": "non_primate_mammals",
    # https://www.genome.jp/virushostdb/318844
    # https://doi.org/10.1016/0378-1135(80)90029-2 antibodies in cattle
    "Tibrogargan virus": "no_mammals",
    # https://www.genome.jp/virushostdb/1272949
    "Manitoba virus": "no_mammals",
    # https://www.genome.jp/virushostdb/1965063
    "DeBrazza's monkey arterivirus": "primate",
    # https://www.genome.jp/virushostdb/998864
    "Alfalfa dwarf virus": "no_mammals",
    # https://www.genome.jp/virushostdb/1965068
    "Simian hemorrhagic encephalitis virus": "primate",
    # https://www.genome.jp/virushostdb/1885929
    "Kibale red colobus virus 1": "primate",
    # https://www.genome.jp/virushostdb/1823757
    "Kafue kinda chacma baboon virus": "primate",
    # https://www.genome.jp/virushostdb/1658615
    "Pebjah virus": "primate",
    # https://www.genome.jp/virushostdb/471728
    "Seal picornavirus type 1": "non_primate_mammals",
    # TODO: verify -- was missing from Aaron's data...
    "avihepatovirus A1": "avian",
    # TODO: verify -- was missing from Aaron's data...
    "Orbivirus alphaequi": "non_primate_mammals",
    # TODO: verify -- was missing from Aaron's data...
    "Kobuvirus bejaponia": "non_primate_mammals",
    # TODO: verify -- was missing from Aaron's data...
    "Potato yellow dwarf virus": "no_mammals",
    # TODO: verify -- was missing from Aaron's data...
    "avian paramyxovirus 12": "avian",
    # TODO: verify -- was missing from Aaron's data...
    "limnipivirus B1": "no_mammals",
                }


def retarget(df, cache_path, n_records=None):
    """
    Parameters
    ----------
    df: pd.DataFrame of training or test data to be relabeled
    cache_path: path to your local cache of the genbank files/records
    n_records: integer number of records to be relabeled (allows
               for partial relabeling, resulting in smaller return
               y/target arrays)
    Returns
    -------
    tuple of arrays with human, mammal, primate labels up to n_records:
    y_human, y_mammal, y_primate
    """
    if n_records is None:
        n_records = df.shape[0]
    y_human = np.empty(shape=(n_records,), dtype=bool)
    y_mammal = np.empty(shape=(n_records,), dtype=bool)
    y_primate = np.empty(shape=(n_records,), dtype=bool)
    idx = 0
    metadata_missing_host = {}
    for row in tqdm(df.iloc[:n_records, ...].itertuples(), desc="Retargeting Data"):
        if row._3: 
            # Human host is always mammal and primate
            y_human[idx] = True
            y_mammal[idx] = True
            y_primate[idx] = True
        else:
            # we need to dig for more information for non-human
            # labeled viruses
            accession = row.Accessions.split()[0]
            genbank_filepath = os.path.join(cache_path, accession, f"{accession}.genbank")
            host_found = False
            with open(genbank_filepath) as genbank_file:
                for record in SeqIO.parse(genbank_file, "genbank"):
                    if "organism" in record.annotations:
                        organism = record.annotations["organism"]
                    # store some relevant metadata for host searching:
                    for feature in record.features:
                        if feature.type == "source":
                            metadata_missing_host = feature.qualifiers
            if not host_found:
                if organism in organism_dict:
                    if organism_dict[organism] == "human":
                        y_human[idx] = True
                        y_mammal[idx] = True
                        y_primate[idx] = True
                        host_found = True
                    elif organism_dict[organism] == "primate":
                        y_human[idx] = False
                        y_mammal[idx] = True
                        y_primate[idx] = True
                        host_found = True
                    elif organism_dict[organism] in {"avian", "no_mammals"}:
                        y_human[idx] = False
                        y_mammal[idx] = False
                        y_primate[idx] = False
                        host_found = True
                    elif organism_dict[organism] == "non_primate_mammals":
                        y_mammal[idx] = True
                        y_primate[idx] = False
                        y_human[idx] = False
                        host_found = True
                    elif organism_dict[organism] == "unknown":
                        # not much is known, categorize as False
                        # for mammals and primates
                        y_human[idx] = False
                        y_mammal[idx] = False
                        y_primate[idx] = False
                        host_found = True
            if not host_found:
                raise ValueError(f"No human host confirmed for {accession=} {organism=}\n"
                                 f"{metadata_missing_host=}")
        idx += 1
    return y_human, y_mammal, y_primate

def main(cache_path):
    df_train = pd.read_csv("viral_seq/data/Mollentze_Training.csv")
    y_human_train, y_mammal_train, y_primate_train = retarget(df=df_train,
                                                              cache_path=cache_path,
                                                              n_records=861)
    df_test = pd.read_csv("viral_seq/data/Mollentze_Holdout.csv")
    y_human_test, y_mammal_test, y_primate_test = retarget(df=df_test,
                                                           cache_path=cache_path,
                                                           n_records=703)
    np.savez("relabeled_data.npz",
             y_human_train=y_human_train,
             y_mammal_train=y_mammal_train,
             y_primate_train=y_primate_train,
             y_human_test=y_human_test,
             y_mammal_test=y_mammal_test,
             y_primate_test=y_primate_test)






if __name__ == "__main__":
    # change this path to the location of your local
    # cache:
    local_cache_path = "/Users/treddy/python_venvs/py_312_host_virus_bioinformatics/lib/python3.12/site-packages/viral_seq/data/cache_viral"
    main(cache_path=local_cache_path)
