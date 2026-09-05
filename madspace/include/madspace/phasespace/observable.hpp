#pragma once

#include "madspace/compgraphs.hpp"
#include "madspace/phasespace/base.hpp"
#include "madspace/util.hpp"

#include <vector>

namespace madspace {

/**
 * Kinematic observable evaluated for a selection of outgoing particles.
 *
 * Selects particles by PDG id (@p pids), optionally groups them into tuples
 * (@p select_pids) and sums the momenta of each tuple, then evaluates one
 * @ref ObservableOption for every selection [1]. Several PDG types may be
 * combined, for example all jet flavours. The available observables are
 * single-momentum functions (`e`, `px`, `py`, `pz`, `mass`, `pt`, `p_mag`,
 * `phi`, `theta`, `y`, `y_abs`, `eta`, `eta_abs`), pair functions
 * (`delta_eta`, `delta_phi`, `delta_r`, and the pair invariant `mass`), and the
 * event-level `sqrt_s`. The same mechanism defines the observables used by
 * @ref Cuts and @ref ObservableHistograms.
 *
 * `batch` is the leading batch dimension. `n` is the number of selected
 * particles and `k` the number of selections.
 *
 * **Arguments**
 * - `momenta` – `float`, shape `(batch, n, 4)` – the selected four-momenta.
 *
 * **Returns**
 * - `observable` – `float`, shape `(batch,)` or `(batch, k)` – the observable
 *   value, one per selection unless @p sum_observable collapses them.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 3.2.4)
 */
class Observable : public FunctionGenerator {
public:
    /// PDG ids commonly grouped as a jet.
    static const std::vector<int> jet_pids;
    /// PDG ids of the b quark.
    static const std::vector<int> bottom_pids;
    /// PDG ids of the charged leptons.
    static const std::vector<int> lepton_pids;
    /// PDG ids treated as missing energy (neutrinos).
    static const std::vector<int> missing_pids;
    /// PDG id of the photon.
    static const std::vector<int> photon_pids;

    /// Kinematic quantity computed by an @ref Observable.
    enum ObservableOption {
        obs_e,
        obs_px,
        obs_py,
        obs_pz,
        obs_mass,
        obs_pt,
        obs_p_mag,
        obs_phi,
        obs_theta,
        obs_y,
        obs_y_abs,
        obs_eta,
        obs_eta_abs,
        obs_delta_eta,
        obs_delta_phi,
        obs_delta_r,
        obs_sqrt_s
    };

    /**
     * @param pids            PDG ids selecting the outgoing particles.
     * @param observable       Which @ref ObservableOption to compute.
     * @param select_pids      Groups of PDG ids forming the tuples the pair /
     *                         summed observables act on; empty selects every
     *                         matching particle individually.
     * @param sum_momenta      Sum the four-momenta within each tuple before
     *                         evaluating the observable.
     * @param sum_observable   Sum the computed observable over all selections.
     * @param order_observable Optional observable used to sort the selections
     *                         before @p order_indices is applied.
     * @param order_indices    After sorting by @p order_observable, keep only
     *                         these positions.
     * @param ignore_incoming  Exclude the two incoming partons from the
     *                         selection.
     * @param name             Label for the returned value.
     */
    Observable(
        const std::vector<int>& pids,
        ObservableOption observable,
        const nested_vector2<int>& select_pids,
        bool sum_momenta = false,
        bool sum_observable = false,
        const std::optional<ObservableOption>& order_observable = std::nullopt,
        const std::vector<int>& order_indices = {},
        bool ignore_incoming = true,
        const std::string& name = ""
    );
    /// The kinematic quantity this observable computes.
    ObservableOption observable() const { return _observable; }
    /// Per-selection lists of particle indices the observable is evaluated on.
    const nested_vector2<me_int_t>& indices() const { return _indices; }
    /// Whether the tuple momenta are summed before evaluation.
    bool sum_momenta() const { return _sum_momenta; }
    /// Flat particle-index list for a single ungrouped, unsummed selection;
    /// empty otherwise.
    std::vector<std::size_t> simple_observable_indices() const {
        if (_sum_momenta || _sum_observable || _indices.size() != 1) {
            return {};
        } else {
            return {_indices.at(0).begin(), _indices.at(0).end()};
        }
    }
    /// The label passed to the constructor.
    std::string name() const { return _name; }
    /// True when no particle in the event matches the requested @p pids.
    bool not_found() const;

private:
    Observable(
        std::tuple<nested_vector2<me_int_t>, nested_vector2<me_int_t>, Type>
            indices_and_type,
        const std::vector<int>& pids,
        ObservableOption observable,
        bool sum_momenta,
        bool sum_observable,
        const std::optional<ObservableOption>& order_observable,
        bool ignore_incoming,
        const std::string& name
    );
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    ObservableOption _observable;
    nested_vector2<me_int_t> _indices;
    std::optional<ObservableOption> _order_observable;
    nested_vector2<me_int_t> _order_indices;
    bool _sum_momenta;
    bool _sum_observable;
    std::string _name;
};

} // namespace madspace
