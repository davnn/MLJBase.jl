module TestPipelines2

using Distributed

# cd(raw"C:\Users\David Muhr\OneDrive - BMW Group\Development\julia\MLJBase\test")
# include("test_utilities.jl")
# include("_models/models.jl")

using MLJBase, MLJModelInterface
using Test
using ..Models
using StableRNGs
using Tables

rng = StableRNG(698790187)

@testset "helpers" begin
    @test MLJBase.individuate([:x, :y, :x, :z, :y, :x]) ==
        [:x, :y, :x2, :z, :y2, :x3]
end

# Dummy Models
lowercase_symbol(s::Symbol) = Symbol(lowercase((string(s))))
for model_name in MLJModelInterface.ABSTRACT_MODEL_SUBTYPES
    my_model_name = Symbol(:My, model_name)
    quote
        mutable struct $my_model_name <: $model_name end
        MLJBase.fit(::$my_model_name, args...) = nothing, nothing, nothing
        MLJBase.transform(::$my_model_name, ::Any, Xnew) = begin
            # @info "Input is $Xnew"
            fill($model_name, nrows(Xnew))
        end
        MLJBase.predict(::$my_model_name, ::Any, Xnew) = begin
            # @info "Input is $Xnew"
            fill($model_name, nrows(Xnew))
        end
    end |> eval
end

transformer_models = [MySupervisedTransformer(),
                      MyUnsupervisedTransformer(),
                      MyStaticTransformer()]

transformer_pipelines = [SupervisedTransformerPipeline,
                         UnsupervisedTransformerPipeline,
                         StaticTransformerPipeline]

probabilistic_models = [MySupervisedProbabilistic(),
                        MyUnsupervisedProbabilistic(),
                        MyStaticProbabilistic()]

probabilistic_pipelines = [SupervisedProbabilisticPipeline,
                           UnsupervisedProbabilisticPipeline,
                           StaticProbabilisticPipeline]

deterministic_models = [MySupervisedDeterministic(),
                        MyUnsupervisedDeterministic(),
                        MyStaticDeterministic()]

deterministic_pipelines = [SupervisedDeterministicPipeline,
                           UnsupervisedDeterministicPipeline,
                           StaticDeterministicPipeline]

interval_models = [MySupervisedInterval(),
                   MyUnsupervisedInterval(),
                   MyStaticInterval()]

interval_pipelines = [SupervisedIntervalPipeline,
                      UnsupervisedIntervalPipeline,
                      StaticIntervalPipeline]

models = vcat(transformer_models,
              probabilistic_models,
              deterministic_models,
              interval_models)

pipelines = vcat(transformer_pipelines,
                 probabilistic_pipelines,
                 deterministic_pipelines,
                 interval_pipelines)

unsupervised_models = [MyUnsupervisedTransformer(),
                       MyUnsupervisedProbabilistic(),
                       MyUnsupervisedDeterministic(),
                       MyUnsupervisedInterval()]

supervised_models = [MySupervisedTransformer(),
                     MySupervisedProbabilistic(),
                     MySupervisedDeterministic(),
                     MySupervisedInterval()]

static_models = [MyStaticTransformer(),
                 MyStaticProbabilistic(),
                 MyStaticDeterministic(),
                 MyStaticInterval()]


suptra, unstra, statra = transformer_models
suppro, unspro, stapro = probabilistic_models
supdet, unsdet, stadet = deterministic_models
supint, unsint, staint = interval_models


NN = 7
X = MLJBase.table(rand(rng, NN, 3));
y = 2X.x1 - X.x2 + 0.05*rand(rng,NN);
Xs = source(X); ys = source(y)

broadcast_mode(v) = mode.(v)
doubler(y) = 2*y

pipe = Pipeline(unstra, unsdet)
pipemach = machine(pipe, X) |> fit!

evaluate(supdet, X, y;
         resampling=Holdout(fraction_train=0.5),
         measures=[accuracy],
         check_measure=false)

UnsupervisedDeterministicPipeline <: Deterministic

# randomly testing various combinations will probably show us more bugs
random_transformer() = rand(transformer_models)
random_probabilistic() = rand(probabilistic_models)
random_deterministic() = rand(deterministic_models)
random_interval() = rand(interval_models)
random_unsupervised() = rand(unsupervised_models)
random_supervised() = rand(supervised_models)
random_static() = rand(static_models)

m = MLJBase.matrix
t = MLJBase.table

@testset "pipe_named_tuple" begin
    @test_throws MLJBase.ERR_EMPTY_PIPELINE MLJBase.pipe_named_tuple((),())
    _names      = (:trf, :fun, :fun, :trf, :clf)
    components = (unstra, m, t, suptra, supdet)
    @test MLJBase.pipe_named_tuple(_names, components) ==
        NamedTuple{(:trf, :fun, :fun2, :trf2, :clf),
                   Tuple{UnsupervisedTransformer,
                         Any,
                         Any,
                         SupervisedTransformer,
                         SupervisedDeterministic}}(components)
end

function test_unnamed_last(component, expect)
    @test Pipeline(m, t,
        random_static(),
        component) isa expect
end

function test_named_last(component, expect)
    @test Pipeline(mat=m, tab=t,
        sta=random_static(),
        las=component) isa expect
end

@testset "last pipeline components leads to expected type" begin
    
    # test last
    for (model, pipeline) in zip(models, pipelines)
        # 1. un-named components
        test_unnamed_last(model, pipeline)
        # 2. named components
        test_named_last(model, pipeline)
    end

    # errors and warnings:
    @test_throws MLJBase.ERR_MIXED_PIPELINE_SPEC Pipeline(m, mymodel=t)
end

@testset "property access" begin
    pipe = Pipeline(m, unstra, unstra, suptra, suptra)

    # property names:
    @test propertynames(pipe) ===
        (:f, :my_unsupervised_transformer, :my_unsupervised_transformer2,
        :my_supervised_transformer, :my_supervised_transformer2, :cache)

    # getindex:
    pipe.my_unsupervised_transformer == unstra
    pipe.my_unsupervised_transformer2 == unstra
    pipe.cache = true

    # replacing a component with one whose abstract supertype is the same
    # or smaller:
    # TODO: What should we allow here?
    # pipe.my_unsupervised_transformer = unstra
    # @test pipe.my_unsupervised_transformer == suptra

    # attempting to replace a component with one whose abstract supertype
    # is bigger:
    # @test_throws MethodError pipe.my_transformer2 = u

    # mutating the components themeselves:
    # pipe.my_unsupervised_transformer.ftr = :z
    # @test pipe.my_unsupervised_transformer.ftr == :z

    # or using MLJBase's recursive getproperty:
    # MLJBase.recursive_setproperty!(pipe, :(my_unsupervised.ftr), :bonzai)
    # @test pipe.my_unsupervised.ftr ==  :bonzai
end

@testset "show" begin
    io = IOBuffer()
    pipe = Pipeline(x-> x^2, m, t, unspro)
    show(io, MIME("text/plain"), pipe)
end

struct TransformerFn <: StaticTransformer
    fn::Function
end

struct ProbabilisticFn <: StaticProbabilistic
    fn::Function
end

struct DeterministicFn <: StaticDeterministic
    fn::Function
end

struct IntervalFn <: StaticInterval
    fn::Function
end

MLJBase.transform(model::TransformerFn, _, args...) = model.fn(args...)
MLJBase.predict(model::Union{ProbabilisticFn,
                             DeterministicFn,
                             IntervalFn}, _, args...) = model.fn(args...)

@testset "pipeline_network_machine" begin
    t = TransformerFn(MLJBase.table)
    m = TransformerFn(MLJBase.matrix)
    f = FeatureSelector()
    h = OneHotEncoder()
    k = KNNRegressor()
    u = UnivariateStandardizer()
    c = ConstantClassifier()

    components = [f, k]
    mach = MLJBase.pipeline_network_machine(
        SupervisedDeterministic, true, components, Xs, ys)
    tree = mach.fitresult.predict |> MLJBase.tree
    @test mach.model isa SupervisedDeterministicSurrogate
    @test tree.operation == predict
    @test tree.model == k
    @test tree.arg1.operation == transform
    @test tree.arg1.model == f
    @test tree.arg1.arg1.source == Xs
    @test tree.arg1.train_arg1.source == Xs
    @test tree.train_arg1 == tree.arg1
    @test tree.train_arg2.source == ys

    components = [f, h]
    mach = MLJBase.pipeline_network_machine(
        UnsupervisedTransformer, true, components, Xs)
    tree = mach.fitresult.transform |> MLJBase.tree
    @test mach.model isa UnsupervisedTransformerSurrogate
    @test tree.operation == transform
    @test tree.model == h
    @test tree.arg1.operation == transform
    @test tree.arg1.model == f
    @test tree.arg1.arg1.source == Xs
    @test tree.arg1.train_arg1.source == Xs
    @test tree.train_arg1 == tree.arg1

    components = [m, t]
    mach = MLJBase.pipeline_network_machine(
        StaticTransformer, true, components, Xs)
    tree = mach.fitresult.transform |> MLJBase.tree
    @test mach.model isa StaticTransformerSurrogate
    @test tree.operation == transform
    @test tree.model == t
    @test tree.arg1.operation == transform
    @test tree.arg1.model == m
    @test tree.arg1.arg1.source == Xs

    # check a probablistic case:
    components = [f, c]
    mach = MLJBase.pipeline_network_machine(
        SupervisedProbabilistic, true, components, Xs, ys)
    @test mach.model isa SupervisedProbabilisticSurrogate

    # check a static case:
    components = [m, t]
    mach = MLJBase.pipeline_network_machine(
        StaticTransformer, true, components, Xs, ys)
    @test mach.model isa StaticTransformerSurrogate

    # An integration test...

    # build a linear network for training:
    components = [f, k]
    mach = MLJBase.pipeline_network_machine(
        SupervisedDeterministic, true, components, Xs, ys)

    # build the same network by hand:
    fM = machine(f, Xs)
    Xt = transform(fM, Xs)
    uM = machine(u, ys)
    yt = transform(uM, ys)
    kM = machine(k, Xt, yt)
    zhat = predict(kM, Xt)
    N2 = inverse_transform(uM, zhat)

    # compare predictions
    fit!(mach, verbosity=0);
    fit!(N2, verbosity=0)
    yhat = predict(mach, X);
    @test yhat ≈ N2()
    k.K = 3; f.features = [:x3,]
    fit!(mach, verbosity=0);
    fit!(N2, verbosity=0)
    @test !(yhat ≈ predict(mach, X))
    @test predict(mach, X) ≈ N2()
    global hand_built = predict(mach, X);

end


# # INTEGRATION TESTS

@testset "integration 1" begin
    # check a simple pipeline prediction agrees with prediction of
    # hand-built learning network built earlier:
    p = Pipeline(FeatureSelector,
                 KNNRegressor)
    p.knn_regressor.K = 3; p.feature_selector.features = [:x3,]
    mach = machine(p, X, y)
    fit!(mach, verbosity=0)
    @test MLJBase.tree(mach.fitresult.predict).model.K == 3
    MLJBase.tree(mach.fitresult.predict).arg1.model.features == [:x3, ]
    @test predict(mach, X) ≈ hand_built

    # test cache is set correctly internally:
    @test all(fitted_params(mach).machines) do m
        MLJBase._cache_status(m)  == " caches data"
    end

    # test correct error thrown for inverse_transform:
    @test_throws(MLJBase.ERR_OP_NOT_SUPPORTED("inverse_transform"),
                 inverse_transform(mach, 3))
end

@testset "integration 2" begin
    # a simple probabilistic classifier pipeline:
    X = MLJBase.table(rand(rng,7,3));
    y = categorical(collect("ffmmfmf"));
    Xs = source(X)
    ys = source(y)
    p = Pipeline(OneHotEncoder, ConstantClassifier, cache=false)
    mach = machine(p, X, y)
    fit!(mach, verbosity=0)
    @test p isa SupervisedProbabilisticComposite
    pdf(predict(mach, X)[1], 'f') ≈ 4/7

    # test cache is set correctly internally:
    @test all(fitted_params(mach).machines) do m
        MLJBase._cache_status(m) == " does not cache data"
    end

    # test invalid replacement of classifier with regressor throws
    # informative error message:
    p.constant_classifier = ConstantRegressor()
    @test_logs((:error, r"^Problem"),
           (:info, r"^Running type"),
           (:warn, r"The scitype of"),
           (:info, r"It seems"),
           (:error, r"Problem"),
               @test_throws Exception fit!(mach, verbosity=-1))
end

@testset "integration 3" begin
    # test a simple deterministic classifier pipeline:
    X = MLJBase.table(rand(rng,7,3))
    y = categorical(collect("ffmmfmf"))
    Xs = source(X)
    ys = source(y)
    p = Pipeline(OneHotEncoder, ConstantClassifier, TransformerFn(broadcast_mode))
    pd = Pipeline(OneHotEncoder, ConstantClassifier, DeterministicFn(broadcast_mode))
    mach = machine(p, X, y)
    machd = machine(pd, X, y)
    fit!(mach, verbosity=0)
    fit!(machd, verbosity=0)
    @test transform(mach, X) == fill('f', 7)
    @test predict(machd, X) == fill('f', 7)

    # test pipelines with weights:
    w = map(y) do η
        η == 'm' ? 100 : 1
    end
    mach = machine(p, X, y, w)
    machd = machine(pd, X, y, w)
    fit!(mach, verbosity=0)
    fit!(machd, verbosity=0)
    @test transform(mach, X) == fill('m', 7)
    @test predict(machd, X) == fill('m', 7)
end

age = [23, 45, 34, 25, 67]
X = (age = age,
     gender = categorical(['m', 'm', 'f', 'm', 'f']))
height = [67.0, 81.5, 55.6, 90.0, 61.1]

mutable struct MyTransformer3 <: StaticTransformer
    ftr::Symbol
end

MLJBase.transform(transf::MyTransformer3, verbosity, X) =
     selectcols(X, transf.ftr)

@testset "integration 4" begin
    #static transformers in pipelines
    p99 = Pipeline(TransformerFn(X -> coerce(X, :age=>Continuous)),
                   OneHotEncoder,
                   MyTransformer3(:age))
    mach = fit!(machine(p99, X), verbosity=0)
    @test transform(mach, X) == float.(X.age)
end

@testset "integration 5" begin
    # pure static pipeline:
    p = Pipeline(TransformerFn(X -> coerce(X, :age=>Continuous)),
                 MyTransformer3(:age))

    mach = fit!(machine(p), verbosity=0) # no training arguments!
    @test transform(mach, X) == X.age

    # and another:
    p = Pipeline(TransformerFn(exp),
                 TransformerFn(log),
                 TransformerFn(x-> 2*x))
    mach = fit!(machine(p), verbosity=0)
    @test transform(mach, 20) ≈ 40
end

@testset "integration 6" begin
    # operation different from predict:
    p = Pipeline(OneHotEncoder,
                 ConstantRegressor)
    @test p isa SupervisedProbabilisticPipeline
    mach = fit!(machine(p, X, height), verbosity=0)
    @test scitype(predict_mean(mach, X)) == AbstractVector{Continuous}
end

MLJBase.supports_inverse_transform(
    ::Union{UnivariateBoxCoxTransformer,UnivariateStandardizer}) = true

@testset "integration 7" begin
    # inverse transform:
    p = Pipeline(UnivariateBoxCoxTransformer,
                 UnivariateStandardizer)
    xtrain = rand(rng, 10)
    mach = machine(p, xtrain)
    fit!(mach, verbosity=0)
    x = rand(rng, 5)
    y = transform(mach, x)
    x̂ = inverse_transform(mach, y)
    @test isapprox(x, x̂)
end

# A dummy clustering model:
mutable struct DummyClusterer <: UnsupervisedTransformer
    n::Int
end

MLJBase.supports_predict(::DummyClusterer) = true

DummyClusterer(; n=3) = DummyClusterer(n)
function MLJBase.fit(model::DummyClusterer, verbosity::Int, X)
    Xmatrix = Tables.matrix(X)
    n = min(size(Xmatrix, 2), model.n)
    centres = Xmatrix[1:n, :]
    levels = categorical(1:n)
    report = (centres=centres,)
    fitresult = levels
    return fitresult, nothing, report
end
MLJBase.transform(model::DummyClusterer, fitresult, Xnew) =
    selectcols(Xnew, 1:length(fitresult))
MLJBase.predict(model::DummyClusterer, fitresult, Xnew) =
    [fill(fitresult[1], nrows(Xnew))...]

@testset "integration 8" begin
    # calling predict on unsupervised pipeline
    # https://github.com/JuliaAI/MLJClusteringInterface.jl/issues/10

    N = 20
    X = (a = rand(N), b = rand(N))

    p = Pipeline(PCA, DummyClusterer)
    mach = machine(p, X)
    fit!(mach, verbosity=0)
    y = predict(mach, X)
    @test y == fill(categorical(1:2)[1], N)
end

@testset "syntactic sugar" begin

    # recall u, s, p, m, are defined way above

    # unsupervised model |> static model:
    pipe1 = unstra |> statra
    @test pipe1 == Pipeline(unstra, statra)

    # unsupervised model |> supervised model:
    pipe2 = unsdet |> suppro
    @test pipe2 == Pipeline(unsdet, suppro)

    # pipe |> pipe:
    hose = pipe1 |> pipe2
    @test hose == Pipeline(unstra, statra, unsdet, suppro)

    # pipe |> model:
    @test Pipeline(unstra, statra) |> suppro == Pipeline(unstra, statra, suppro)

    # model |> pipe:
    @test unstra |> Pipeline(statra, suppro) == Pipeline(unstra, statra, suppro)

    # pipe |> function:
    @test Pipeline(unsdet, statra) |> TransformerFn(m) == Pipeline(unsdet, statra, TransformerFn(m))

    # function |> pipe:
    @test TransformerFn(m) |> Pipeline(statra, suppro) == Pipeline(TransformerFn(m), statra, suppro)

    # model |> function:
    @test unsdet |> TransformerFn(m) == Pipeline(unsdet, TransformerFn(m))

    # function |> model:
    @test TransformerFn(t) |> unsdet == Pipeline(TransformerFn(t), unsdet)

    @test_logs((:info, MLJBase.INFO_AMBIGUOUS_CACHE),
               Pipeline(unsdet, cache=false) |> suppro)

    # with types
    @test PCA |> Standardizer() |> KNNRegressor ==
        Pipeline(PCA(), Standardizer(), KNNRegressor())
end

end

true