#pragma once

#include "madspace/driver/backend.hpp"
#include "madspace/driver/context.hpp"
#include "madspace/phasespace/base.hpp"

namespace madspace {

class GradientClipper : public FunctionGenerator {
public:
    GradientClipper();

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;
};

class AdamOptimizer {
public:
    enum LRSchedule {
        none,
        cosine,
    };

    AdamOptimizer(
        const Function& function,
        ContextPtr context,
        double learning_rate,
        LRSchedule schedule = LRSchedule::none,
        std::size_t step_count = 0,
        double beta1 = 0.9,
        double beta2 = 0.999,
        double eps = 1e-8,
        double grad_clip_threshold = 0.0,
        double weight_decay = 0.0
    );
    TensorVec step(const TensorVec& inputs);
    void replace_function(const Function& function);
    double learning_rate() const;
    const TypeVec& input_types() const { return _input_types; }
    ContextPtr context() const { return _context; }
    Tensor parameters() const { return _parameter; }
    const std::vector<std::string>& param_names() const { return _param_names; }

private:
    ContextPtr _context;
    RuntimePtr _runtime;
    LRSchedule _schedule;
    double _learning_rate;
    std::size_t _step;
    std::size_t _step_count;
    double _beta1;
    double _beta2;
    double _eps;
    double _grad_clip_threshold;
    double _weight_decay;
    double _loss_mean;
    RuntimePtr _grad_clipper;
    Tensor _one;
    Tensor _parameter;
    Tensor _exp_avg;
    Tensor _exp_avg_sq;
    Tensor _threshold_tensor;
    TypeVec _input_types;
    std::vector<std::string> _param_names;
};

} // namespace madspace
