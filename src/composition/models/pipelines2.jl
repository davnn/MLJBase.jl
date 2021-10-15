# Code to construct pipelines without macros

# ## Note on mutability.

# The components in a pipeline, as defined here, can be replaced so
# long as their "abstract supertype" (eg, `Probabilistic`) remains the
# same. This is the type returned by `abstract_type()`; in the present
# code it will always be one of the types listed in
# `SUPPORTED_TYPES_FOR_PIPELINES` below, or `Any`, if `component` is
# not a model (which, by assumption, means it is callable).


# # HELPERS

# modify collection of symbols to guarantee uniqueness. For example,
# individuate([:x, :y, :x, :x]) = [:x, :y, :x2, :x3])
function individuate(v)
    isempty(v) && return v
    ret = Symbol[first(v),]
    for s in v[2:end]
        s in ret || (push!(ret, s); continue)
        n = 2
        candidate = s
        while true
            candidate = string(s, n) |> Symbol
            candidate in ret || break
            n += 1
        end
        push!(ret, candidate)
    end
    return ret
end

# extend uppercasefirst for symbols
uppercasefirst(s::Symbol) = Symbol(uppercasefirst(string(s)))

_instance(x) = x
_instance(T::Type{<:Model}) = T()


# # TYPES

const SUPPORTED_TYPES_FOR_PIPELINES = MLJModelInterface.ABSTRACT_MODEL_SUBTYPES

const pipeline_type_given_type(type::Symbol) = Symbol(type, :Pipeline)
const composite_type_given_type(type::Symbol) = Symbol(type, :Composite)

const PREDICTION_TYPE_OPTIONS = [:deterministic,
                                 :probabilistic,
                                 :interval]

for T_ex in SUPPORTED_TYPES_FOR_PIPELINES
    P_ex = pipeline_type_given_type(T_ex)
    C_ex = composite_type_given_type(T_ex)
    quote
        mutable struct $P_ex{N<:NamedTuple} <: $C_ex
            named_components::N
            cache::Bool
            $P_ex(named_components::N, cache) where N =
                new{N}(named_components, cache)
        end
    end |> eval
end

# hack an alias for the union type, `SomePipeline{N}`:
const PIPELINE_TYPES = pipeline_type_given_type.(SUPPORTED_TYPES_FOR_PIPELINES)
const _TYPE_EXS = map(PIPELINE_TYPES) do P_ex
    Meta.parse("$(P_ex){N}")
end
quote
    const SomePipeline{N} =
        Union{$(_TYPE_EXS...)}
end |> eval

components(p::SomePipeline) = values(getfield(p, :named_components))
names(p::SomePipeline) = keys(getfield(p, :named_components))

# # GENERIC CONSTRUCTOR

const PRETTY_PREDICTION_OPTIONS =
    join([string("`:", opt, "`") for opt in PREDICTION_TYPE_OPTIONS],
         ", ",
         " and ")
const ERR_TOO_MANY_SUPERVISED = ArgumentError(
    "More than one supervised model in a pipeline is not permitted")
const ERR_EMPTY_PIPELINE = ArgumentError(
    "Cannot create an empty pipeline. ")
const ERR_INVALID_PREDICTION_TYPE = ArgumentError(
    "Invalid `prediction_type` encountered.")
const ERR_MIXED_PIPELINE_SPEC = ArgumentError(
    "Either specify all pipeline components without names, as in "*
    "`Pipeline(model1, model2)` or all specify names for all "*
    "components, as in `Pipeline(myfirstmodel=model1, mysecondmodel=model2)`. ")

# The following combines its arguments into a named tuple, performing
# a number of checks and modifications. Specifically, it checks
# `components` as a is valid sequence, modifies `names` to make them
# unique, and replaces the types appearing in the named tuple type
# parameters with their abstract supertypes. See the "Note on
# mutability" above.
function pipe_named_tuple(names, components)

    isempty(names) && throw(ERR_EMPTY_PIPELINE)

    # make keys unique:
    names = names |> individuate |> Tuple

    # return the named tuple:
    types = abstract_type.(components)
    NamedTuple{names,Tuple{types...}}(components)
end

"""
    Pipeline(component1, component2, ... , componentk; options...)
    Pipeline(name1=component1, name2=component2, ..., namek=componentk; options...)
    component1 |> component2 |> ... |> componentk

Create an instance of composite model type which sequentially composes
the specified components in order. This means `component1` receives
inputs, whose output is passed to `component2`, and so forth. A
"component" is either a `Model` instance, a model type (converted
immediately to its default instance) or any callable object. Here the
"output" of a model is what `predict` returns if it is `Supervised`,
and what `transform` returns if it is `Unsupervised`.

Names for the component fields are automatically generated unless
explicitly specified, as in

```
Pipeline(endoder=ContinuousEncoder(drop_last=false),
         stand=Standardizer())
```

The `Pipeline` constructor accepts key-word `options` discussed further
below.

Ordinary functions (and other callables) may be inserted in the
pipeline as shown in the following example:

    Pipeline(X->coerce(X, :age=>Continuous), OneHotEncoder, ConstantClassifier)

### Syntactic sugar

The `|>` operator is overloaded to construct pipelines out of models,
callables, and existing pipelines:

```julia
LinearRegressor = @load LinearRegressor pkg=MLJLinearModels add=true
PCA = @load PCA pkg=MultivariateStats add=true

pipe1 = MLJBase.table |> ContinuousEncoder |> Standardizer
pipe2 = PCA |> LinearRegressor
pipe1 |> pipe2
```

At most one of the components may be a supervised model, but this
model can appear in any position. A pipeline with a `Supervised`
component is itself `Supervised` and implements the `predict`
operation.  It is otherwise `Unsupervised` (possibly `Static`) and
implements `transform`.

### Special operations

If all the `components` are invertible unsupervised models
(transformers), then `inverse_transform` is implemented for the
pipeline. If there are no supervised models, then `predict` is
nevertheless implemented, assuming the last component is a model that
implements it (some clustering models). Similarly, calling `transform`
on a supervised pipeline calls `transform` on the supervised
component.

### Optional key-word arguments

- `prediction_type`  -
  prediction type of the pipeline; possible values: `:deterministic`,
  `:probabilistic`, `:interval` (default=`:deterministic` if not inferable)

- `operation` - operation applied to the supervised component model,
  when present; possible values: `predict`, `predict_mean`,
  `predict_median`, `predict_mode` (default=`predict`)

- `cache` - whether the internal machines created for component models
  should cache model-specific representations of data (see
  [`machine`](@ref)) (default=`true`)

!!! warning

    Set `cache=false` to guarantee data anonymization.

To build more complicated non-branching pipelines, refer to the MLJ
manual sections on composing models.

"""
function Pipeline(args...;
                  cache=true,
                  kwargs...)

    # Components appear either as `args` (with names to be
    # automatically generated) or in `kwargs`, but not both.

    # This public constructor does checks and constructs a valid named
    # tuple, `named_components`, to be passed onto a secondary
    # constructor.

    isempty(args) || isempty(kwargs) ||
        throw(ERR_MIXED_PIPELINE_SPEC)

    # construct the named tuple of components:
    if isempty(args)
        _names = keys(kwargs)
        _components = values(values(kwargs))
    else
        _names = Symbol[]
        for c in args
            generate_name!(c, _names, only=Model)
        end
        _components = args
    end

    # in case some components are specified as model *types* instead
    # of instances:
    components = _instance.(_components)

    named_components = pipe_named_tuple(_names, components)

    _pipeline(named_components, cache)
end

function _pipeline(named_components::NamedTuple, cache)

    # This method assumes all arguments are valid and includes the
    # logic that determines which concrete pipeline's constructor
    # needs calling.

    components = values(named_components)

    # The pipeline is supervised if it contains at least one
    # supervised component
    idx = findfirst(components) do c
        typeof(c) <: Supervised
    end
    is_supervised = idx !== nothing

    # The pipeline is static if it contains only static components 
    static_components = filter(components) do m
        !(m isa Model) || typeof(m) <: Static
    end
    is_static = length(static_components) == length(components)

    # determine the prediction type based on the last pipeline component
    raw_pred_type = prediction_type(typeof(last(components)))
    pred_type = raw_pred_type == :unknown ? :Transformer :
                raw_pred_type == :probabilistic ? :Probabilistic :
                raw_pred_type == :deterministic ? :Deterministic :
                raw_pred_type == :interval ? :Interval :
                throw(ERR_INVALID_PREDICTION_TYPE)

    # To make final pipeline type determination, we need to determine
    # the corresonding abstract type (eg, `Probablistic`) here called
    # `super_type`:
    super_type = (is_supervised ? Symbol(:Supervised, pred_type) :
                  is_static ? Symbol(:Static, pred_type) :
                  Symbol(:Unsupervised, pred_type)) |> eval

    # dispatch on `super_type` to construct the appropriate type:
    _pipeline(super_type, named_components, cache)
end

# where the method called in the last line will be one of these:
for T_ex in SUPPORTED_TYPES_FOR_PIPELINES
    P_ex = pipeline_type_given_type(T_ex)
    quote
        _pipeline(::Type{<:$T_ex}, args...) =
            $P_ex(args...)
    end |> eval
end


# # PROPERTY ACCESS

err_pipeline_bad_property(p, name) = ErrorException(
    "type $(typeof(p)) has no property $name")

Base.propertynames(p::SomePipeline{<:NamedTuple{names}}) where names =
    (names..., :cache)

function Base.getproperty(p::SomePipeline{<:NamedTuple{names}},
                          name::Symbol) where names
    name === :cache && return getfield(p, :cache)
    name in names && return getproperty(getfield(p, :named_components), name)
    throw(err_pipeline_bad_property(p, name))
end

function Base.setproperty!(p::SomePipeline{<:NamedTuple{names,types}},
                           name::Symbol, value) where {names,types}
    name === :cache && return setfield!(p, :cache, value)
    idx = findfirst(==(name), names)
    idx === nothing && throw(err_pipeline_bad_property(p, name))
    components = getfield(p, :named_components) |> values |> collect
    @inbounds components[idx] = value
    named_components = NamedTuple{names,types}(Tuple(components))
    setfield!(p, :named_components, named_components)
end


# # LEARNING NETWORK MACHINES FOR PIPELINES

# https://alan-turing-institute.github.io/MLJ.jl/dev/composing_models/#Learning-network-machines

#
#             StaticT      UnsupervisedT         StaticP
# Pipeline(Standardizer(), KNNDetector(), ProbabilisticDetector())
#
# Xs, ys = source(X), source(y)
# Xstd = transform(machine(Standardizer(), Xs))
# scores = transform(machine(KNNDetector(), Xstd))
# result = predict(machine(ProbabilisticDetector(), scores))

function to_machine(component, next_input, cache, sources...)
    mach = typeof(component) <: Static ? machine(component, cache=cache) :
           machine(component, next_input, sources...; cache=cache)
    return mach
end

function extend(prev_result, next_component, cache, sources...)
    prev_component, prev_mach, prev_input = prev_result
    op = typeof(prev_component) <: Transformer ? transform : predict
    next_input = op(prev_mach, prev_input)
    next_mach = to_machine(next_component, next_input, cache, sources...)
    return (next_component, next_mach, next_input)
end

# ## The learning network machine

const ERR_OP_NOT_SUPPORTED(op) = ErrorException(
    "Applying `$op` to a pipeline "*
    "that does not support it")

# helper traits that should be in MLJ, or are there already?
supports_transform(::Transformer) = true
supports_transform(::Any) = false

supports_predict(::Union{Probabilistic, Deterministic, Interval}) = true
supports_predict(::Any) = false

supports_predict_mode(::Probabilistic) = true
supports_predict_mode(::Any) = false

supports_predict_mean(::Probabilistic) = true
supports_predict_mean(::Any) = false

supports_predict_median(::Probabilistic) = true
supports_predict_median(::Any) = false

supports_predict_joint(::Any) = false
supports_inverse_transform(::Any) = false

# determine if a component supports operations and 
# return the corresponding nodes if it does
function get_nodes(machine, component, source0)
    ops = [:transform, :predict, :predict_mode, :predict_mean, :predict_median, :predict_joint]
    nodes = []

    for op in ops
        supports = Symbol(:supports_, op)
        quote
            push!($nodes, $supports($component) ?
                $op($machine, $source0) :
                ErrorNode(ERR_OP_NOT_SUPPORTED($op)))
        end |> eval
    end

    return NamedTuple{Tuple(ops)}(nodes)
end

function pipeline_network_machine(super_type,
                                  cache,
                                  components,
                                  source0,
                                  sources...)

    # create the extend closure
    _extend(prev, next) = extend(prev, next, cache, sources...)

    # create the initial result
    comp0, compn = first(components), components[2:end]
    mach0 = to_machine(comp0, source0, cache, sources...)
    init0 = (comp0, mach0, source0)

    # reduce all intermediate results
    final_component, final_machine, final_input = foldl(_extend, compn, init=init0)

    # get all trait-based available nodes
    nodes = get_nodes(final_machine, final_component, final_input)

    # backwards pass to get `inverse_transform` node
    if all(c -> supports_inverse_transform(c), components)
        inode = source0
        node = nodes.transform
        for _ in eachindex(components)
            mach = node.machine
            inode = inverse_transform(mach, inode)
            node =  first(mach.args)
        end
    else
        inode = ErrorNode(ERR_OP_NOT_SUPPORTED("inverse_transform"))
    end

    # create the final surrogate machine
    machine(super_type(), source0, sources...; inverse_transform=inode, nodes...)
end

# # FIT METHOD

function MMI.fit(pipe::SomePipeline{N},
                 verbosity::Integer,
                 arg0=source(),
                 args...) where {N}

    source0 = source(arg0)
    sources = source.(args)

    _components = components(pipe)

    mach = pipeline_network_machine(abstract_type(pipe),
                                    pipe.cache,
                                    _components,
                                    source0,
                                    sources...)
    return!(mach, pipe, verbosity)
end


# # SYNTACTIC SUGAR

const INFO_AMBIGUOUS_CACHE =
    "Joining pipelines with conflicting `cache` values. Using `cache=false`. "

import Base.(|>)
const compose = (|>)

Pipeline(p::SomePipeline) = p

const FuzzyModel = Union{Model,Type{<:Model}}

function compose(m1::FuzzyModel, m2::FuzzyModel)

    # no-ops for pipelines:
    p1 = Pipeline(m1)
    p2 = Pipeline(m2)

    _components = (components(p1)..., components(p2)...)
    _names = (names(p1)..., names(p2)...)

    named_components = pipe_named_tuple(_names, _components)

    # `cache` is only `true` if `true` for both pipelines:
    cache = false
    if p1.cache && p2.cache
        cache = true
    elseif p1.cache ⊻ p2.cache
        @info INFO_AMBIGUOUS_CACHE
    end

    _pipeline(named_components, cache)
end

compose(p1, p2::FuzzyModel) = compose(Pipeline(p1), p2)
compose(p1::FuzzyModel, p2) = compose(p1, Pipeline(p2))

# export all pipeline types
for T in PIPELINE_TYPES
    @eval(export $T)
end
