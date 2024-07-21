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
                                                              n_records=215)
    df_test = pd.read_csv("viral_seq/data/Mollentze_Holdout.csv")
    y_human_test, y_mammal_test, y_primate_test = retarget(df=df_test,
                                                           cache_path=cache_path,
                                                           n_records=200)
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
