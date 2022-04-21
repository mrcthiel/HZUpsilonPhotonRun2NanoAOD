import hist
from coffea import processor

from samples import samples_files
from HZUpsilonPhotonRun2NanoAOD.hist_accumulator import HistAccumulator
from HZUpsilonPhotonRun2NanoAOD.sample_processor import SampleProcessor


class Analyzer(processor.ProcessorABC):
    def __init__(self):
        self._accumulator = processor.dict_accumulator(
            {
                "cutflow": HistAccumulator(
                    hist.Hist.new.StrCat(samples_files.keys(), name="dataset")
                    .StrCat(["2016APV", "2016", "2017", "2018"], name="year")
                    .Bool(name="trigger")
                    .Bool(name="nmuons")
                    .Bool(name="muon_pt")
                    .Bool(name="mediumPrompt_muon")
                    .Bool(name="iso_muon")
                    .Bool(name="nphotons")
                    .Bool(name="photon_pt")
                    .Bool(name="photon_sc_eta")
                    .Bool(name="photon_electron_veto")
                    .Bool(name="photon_tight_id")
                    .Bool(name="signal_selection")
                    .Bool(name="mass_selection")
                    .Double()
                ),
            }
        )

    @property
    def accumulator(self):
        return self._accumulator

    # we will receive NanoEvents
    def process(self, events):

        output = self.accumulator.identity()
        processor = SampleProcessor(events, output)
        return processor()

    def postprocess(self, accumulator):
        return accumulator
