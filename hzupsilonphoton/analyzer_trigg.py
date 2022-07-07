import awkward as ak
from coffea import processor
from coffea.processor import Accumulatable, defaultdict_accumulator, dict_accumulator

from hzupsilonphoton.events import Events
from hzupsilonphoton.forward_events import forward_events
from hzupsilonphoton.utils import save_events_trigg


class Analyzer_Trigg(processor.ProcessorABC):  # type: ignore
    def __init__(self) -> None:
        self._accumulator = dict_accumulator({})

    @property
    def accumulator(self) -> Accumulatable:
        return self._accumulator

    # we will receive NanoEvents
    def process(self, events: ak.Array) -> Accumulatable:

        # Forward events over the defined analysis workflow
        evts = forward_events(Events(events))

        # Save kinematical information of selected events
        save_events_trigg(
            evts=evts,
            prefix="selected_events",
            list_of_filters=[
                "lumisection",
                "trigger",
                "n_muons",
                "n_photons",
                "n_dimuons",
                "n_bosons",
                "signal_selection",
                "mass_selection",
            ],
        )

        return self.accumulator

    def postprocess(self, accumulator: Accumulatable) -> Accumulatable:
        return accumulator
