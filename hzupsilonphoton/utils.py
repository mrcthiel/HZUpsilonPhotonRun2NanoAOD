import secrets
from typing import Union

import awkward as ak
import numpy as np
import uproot
from coffea.nanoevents.methods.candidate import Candidate
from coffea.processor import Accumulatable
from numpy.typing import ArrayLike
from particle import PDGID, Particle

from hzupsilonphoton.events import Events


def file_tester(file_path: str) -> None:
    try:
        uproot.open(file_path).close()
    except Exception:
        print(f"An exception occurred trying to open: {file_path}")


def safe_mass(candidate: Candidate) -> ArrayLike:
    """Get the mass of a canditate, taking care of negative mass**2 due to NanoAOD precision issues."""
    squared_mass = candidate.mass2
    return np.sqrt(ak.where(squared_mass < 0, 0, squared_mass))


def get_pdgid_by_name(name: str) -> PDGID:
    return Particle.from_name(name).pdgid


def mc_sample_filter(dataset: str, events: ak.Array) -> Union[ArrayLike, ak.Array]:
    """Filter MC samples for special cases."""
    _filter = np.ones(len(events), dtype=bool)

    # Higss resonant m_ll < 30
    if dataset.startswith(
        "GluGluHToMuMuG_M125_MLL-0To60_Dalitz_012j_13TeV_amcatnloFXFX_pythia8"
    ):
        is_prompt_filter = events.GenPart.hasFlags("isPrompt")
        is_mu_plus_filter = events.GenPart.pdgId == get_pdgid_by_name("mu+")
        is_mu_minus_filter = events.GenPart.pdgId == get_pdgid_by_name("mu-")
        # is_gamma_filter = events.GenPart.pdgId == get_pdgid_by_name("gamma")
        is_Higgs_children = ak.fill_none(
            events.GenPart.parent.pdgId == get_pdgid_by_name("H0"), False
        )

        muons_plus = events.GenPart[
            is_prompt_filter & is_Higgs_children & is_mu_plus_filter
        ]
        muons_minus = events.GenPart[
            is_prompt_filter & is_Higgs_children & is_mu_minus_filter
        ]

        dimuons_masses = ak.flatten(safe_mass(muons_plus + muons_minus))
        dimuons_masses_filter = dimuons_masses < 30
        _filter = dimuons_masses_filter

    # Z signal m_ll > 50 (? - Not sure if it should be done)
    # if dataset.startswith("ZToUpsilon"):
    #     pass
    return _filter


def save_dimuon_masses(evts: Events, list_of_dimuons_mass_filters: list[str]) -> None:

    dimuons = evts.events.dimuons[
        evts.filters.all(*list_of_dimuons_mass_filters),
    ]
    dimuons_mass = safe_mass(dimuons["0"] + dimuons["1"])

    dimuons_mass_filename = f"outputs/buffer/dimuons_mass_{evts.dataset}_{evts.year}_{secrets.token_hex(nbytes=20)}.root"
    with uproot.recreate(dimuons_mass_filename) as f:
        f["dimuons_masses"] = {"mass": ak.flatten(dimuons_mass)}


def save_events(evts: Events, prefix: str, list_of_filters: list[str]) -> None:
    """Save kinematical information of selected events."""
    selection_filter = evts.filters.all(*list_of_filters)
    selected_events = evts.events[selection_filter]

    boson = selected_events.boson
    upsilon = selected_events.upsilon
    photon = selected_events.photon
    mu_1 = selected_events.mu_1
    mu_2 = selected_events.mu_2

    output_filename = f"outputs/buffer/{prefix}_{evts.dataset}_{evts.year}_{secrets.token_hex(nbytes=20)}.root"
    buffer = {
        "boson_mass": ak.flatten(safe_mass(boson)),
        "boson_pt": ak.flatten(boson.pt),
        "boson_eta": ak.flatten(boson.eta),
        "boson_phi": ak.flatten(boson.phi),
        "upsilon_mass": ak.flatten(safe_mass(upsilon)),
        "upsilon_pt": ak.flatten(upsilon.pt),
        "upsilon_eta": ak.flatten(upsilon.eta),
        "upsilon_phi": ak.flatten(upsilon.phi),
        "photon_mass": ak.flatten(photon.mass),
        "photon_pt": ak.flatten(photon.pt),
        "photon_eta": ak.flatten(photon.eta),
        "photon_phi": ak.flatten(photon.phi),
        "mu_1_mass": ak.flatten(mu_1.mass),
        "mu_1_pt": ak.flatten(mu_1.pt),
        "mu_1_eta": ak.flatten(mu_1.eta),
        "mu_1_phi": ak.flatten(mu_1.phi),
        "mu_2_mass": ak.flatten(mu_2.mass),
        "mu_2_pt": ak.flatten(mu_2.pt),
        "mu_2_eta": ak.flatten(mu_2.eta),
        "mu_2_phi": ak.flatten(mu_2.phi),
        "delta_eta_upsilon_photon": ak.flatten(np.absolute(upsilon.eta - photon.eta)),
        "delta_phi_upsilon_photon": ak.flatten(np.absolute(upsilon.delta_phi(photon))),
        "delta_r_upsilon_photon": ak.flatten(upsilon.delta_r(photon)),
        "weight": evts.weights.weight()[selection_filter],
    }
    for w in evts.weights.names:
        buffer[f"weight_{w}"] = evts.weights.individual_weight(w)[selection_filter]

    with uproot.recreate(output_filename) as f:
        f["Events"] = buffer


def two_powers(num):
    return [ 1 << idx for idx in range(num.bit_length()) if num & (1 << idx) ]

def save_events_trigg(evts: Events, prefix: str, list_of_filters: list[str]) -> None:
    """Save kinematical information of selected events."""
    selection_filter = evts.filters.all(*list_of_filters)
    selected_events = evts.events[selection_filter]

    probe_photon = selected_events.probe_photon
    probe_muon = selected_events.probe_muon
    tag_muon = selected_events.tag_muon 
    TrigObjs = selected_events.TrigObjs

    output_filename = f"outputs_trigg/buffer/{prefix}_{evts.dataset}_{evts.year}_{secrets.token_hex(nbytes=20)}.root"

    # GOOD PROBE MUON

'''
testing by hand:
import uproot
import awkward as ak
import numpy as np
f = uproot.open('step0NANOAOD_trg.root:Events')
eta = (ak.cartesian([f['Muon_eta'].arrays()['Muon_eta'],f['TrigObj_eta'].arrays()['TrigObj_eta']], nested=True))
delta = abs(eta["0"]-eta["1"])
delta = ak.flatten(delta[ak.argsort(delta, ascending=True)][:,:,:1], axis=2)
'''

    deltaR_probe_muon_combinations = ak.cartesian([probe_muon,TrigObjs], nested=True)
    deltaR_probe_muon_all = (deltaR_probe_muon_combinations["0"]).delta_r(deltaR_probe_muon_combinations["1"])
    deltaR_probe_muon_sort = ak.flatten(deltaR_probe_muon_all[ak.argsort(deltaR_probe_muon_all, ascending=True)][:,:,:1], axis=2) #retornando so menor dr para cada muon 
    TrigObj_probe_muon_id = ak.flatten(TrigObjs.id[[ak.argsort(deltaR_probe_muon_all, ascending=True)[:,:,:1]], axis=2) == 13
    TrigObj_probe_muon_filterBits = two_powers(ak.flatte(TrigObjs.filterBits[ak.argsort(deltaR_probe_muon_all, ascending=True)[:,:,:1]], axis=2)))
    TrigObj_probe_muon_filterBits_bool = [x in [32] for x in TrigObj_probe_muon_filterBits]  
    good_probe_muon = deltaR_probe_muon_sort < 0.1 && TrigObj_probe_muon_id && TrigObj_probe_muon_filterBits_bool


    buffer = {
#        "probe_photon_mass": ak.flatten(probe_photon.mass),
#        "probe_photon_pt": ak.flatten(probe_photon.pt),
#        "probe_photon_eta": ak.flatten(probe_photon.eta),
#        "probe_photon_phi": ak.flatten(probe_photon.phi),
#        "tag_muon_mass": ak.flatten(tag_muon.mass),
#        "tag_muon_pt": ak.flatten(tag_muon.pt),
#        "tag_muon_eta": ak.flatten(tag_muon.eta),
#        "tag_muon_phi": ak.flatten(tag_muon.phi),
        "probe_muon_mass": ak.flatten(probe_muon.mass),
        "probe_muon_pt": ak.flatten(probe_muon.pt),
        "probe_muon_eta": ak.flatten(probe_muon.eta),
        "probe_muon_phi": ak.flatten(probe_muon.phi),
#        "weight": evts.weights.weight()[selection_filter],
#	"triplet_weight": ?? 
#        "good_probe": ak.flatten(good_probe_muon and good_probe_photon)
        "good_probe_muon": ak.flatten(good_probe_muon)
#        "good_probe_photon": ak.flatten(TrigObj_id_photon and deltaR_photon and filterBits_photon)
    }
    for w in evts.weights.names:
        buffer[f"weight_{w}"] = evts.weights.individual_weight(w)[selection_filter]

    with uproot.recreate(output_filename) as f:
        f["Events"] = buffer


def fill_cutflow(
    accumulator: Accumulatable,
    evts: Events,
    key: str,
    variation: str,
    list_of_weights: list[str],
    list_of_filters: list[str],
) -> None:
    accumulator[key][
        f"{evts.dataset}_{evts.year}"
    ] = evts.weights.partial_weight_with_variation(
        variation_name=variation, include=list_of_weights
    )[
        evts.filters.all(*list_of_filters)
    ].sum()
