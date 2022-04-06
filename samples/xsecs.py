# References: https://docs.google.com/spreadsheets/d/1zvvedRr1K4NFylNqxNdkWbFgoMPvh9E9BrqxywBpu-Y/edit#gid=0

xsecs = {}
xsecs['ggH_HToUps1SG_M125_NNPDF31_TuneCP5_13TeV-powheg-pythia8'] = 1 # in pb
xsecs['ggH_HToUps2SG_M125_NNPDF31_TuneCP5_13TeV-powheg-pythia8'] = 1 # in pb
xsecs['ggH_HToUps3SG_M125_NNPDF31_TuneCP5_13TeV-powheg-pythia8'] = 1 # in pb
xsecs['GluGluHToMuMuG_M125_MLL-0To60_Dalitz_012j_13TeV_amcatnloFXFX_pythia8_PSWeight'] = 1 # in pb
xsecs['ZGTo2MuG_MMuMu-2To15_TuneCP5_13TeV-madgraph-pythia8'] = 1 # in pb
xsecs['ZToUpsilon1SGamma_TuneCP5_13TeV-amcatnloFXFX-pythia8'] = 1 # in pb
xsecs['ZToUpsilon2SGamma_TuneCP5_13TeV-amcatnloFXFX-pythia8'] = 1 # in pb
xsecs['ZToUpsilon3SGamma_TuneCP5_13TeV-amcatnloFXFX-pythia8'] = 1 # in pb

def x_section(dataset):
    for xs in xsecs:
        if dataset.startswith(xs):
            return xsecs[xs]
