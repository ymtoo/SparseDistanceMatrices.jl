using Distances, NearestNeighbors, SparseDistanceMatrices, BenchmarkTools

using Test

@testset "SparseDistanceMatrices" begin

    N = 100
    Ts = [Int16, Int32, Int64, Float16, Float32, Float64]

    for T in Ts
        D = SparseDistanceMatrix(N, Int[], Int[], T[])
        @test D == SparseDistanceMatrix(N, T)
        a = T <: Integer ? trunc(T, 10.0) : T(10.0)
        b = T <: Integer ? trunc(T, 12.3) : T(12.3)
        D[1,20] = a
        D[40,2] = b

        @test size(D) == (100, 100)
        [@test D[i,i] == zero(T) for i in 1:N]
        @test D[1,100] == typemax(T)
        @test D[1,20] == a
        @test D[40,2] == b
        @test sum(D .!== typemax(T)) == 2+N
        @test D != D'
        @test symmetrize!(D) == D'
        @test countnt(D) == 4

        am = adjacency_matrix(D)
        @test am[10,11] == am[50,51] == am[90,91] == zero(T)

        ndval = T <: Integer ? trunc(T, 1000.0) : T(1000.0)
        D = SparseDistanceMatrix(N, Int[], Int[], T[], ndval)
        @test D == SparseDistanceMatrix(N, ndval)
        a = T <: Integer ? trunc(T, 10.0) : T(10.0)
        b = T <: Integer ? trunc(T, 12.3) : T(12.3)
        D[1,20] = a
        D[40,2] = b

        @test size(D) == (100, 100)
        [@test D[i,i] == zero(T) for i in 1:N]
        @test D[1,100] == ndval
        @test D[1,20] == a
        @test D[40,2] == b
        @test sum(D .!== ndval) == 2+N
        @test D != D'
        @test symmetrize!(D) == D'
        @test countnt(D) == 4

        am = adjacency_matrix(D)
        @test am[10,11] == am[50,51] == am[90,91] == zero(T)

        # check double entry
        D[40,2] = a
        @test D[40,2] == a
        @test length(D.rowindices) == length(D.colindices) == length(D.ndval) == 4
        @test countnt(D) == 4

        @test_throws ArgumentError D[1,1] = T <: Integer ? trunc(T, 10.0) : T(10.0)
    end

end

@testset "Pairwise" begin
    X = [0 0 1 1 1;
         1 0 1 0 1;
         0 1 1 1 1;
         0 0 1 1 0]
    D1 = pairwise(Euclidean(), X, 4; dims=2)
    Dt1 = pairwise(Euclidean(), transpose(X), 4; dims=1)
    @test D1 == Dt1 == [0.0 Inf Inf Inf Inf;
                        Inf 0.0 Inf Inf Inf;
                        Inf Inf 0.0 1.0 1.0;
                        Inf Inf 1.0 0.0 Inf;
                        Inf Inf 1.0 Inf 0.0]
    D2 = pairwise(Euclidean(), X, 1; dims=2)
    Dt2 = pairwise(Euclidean(), transpose(X), 1; dims=1)
    @test D2 == Dt2 == [0.0 Inf Inf Inf Inf;
                        Inf 0.0 Inf Inf Inf;
                        Inf Inf 0.0 1.0 Inf;
                        Inf Inf Inf 0.0 Inf;
                        Inf Inf Inf Inf 0.0]
    D3 = pairwise(Euclidean(), X, 2; dims=2)
    Dt3 = pairwise(Euclidean(), transpose(X), 2; dims=1)
    @test D3 == Dt3 == [0.0 Inf Inf Inf Inf;
                        Inf 0.0 Inf Inf Inf;
                        Inf Inf 0.0 1.0 1.0;
                        Inf Inf Inf 0.0 Inf;
                        Inf Inf Inf Inf 0.0]

    Xf = Float32.([0.1 0.1 1.0 1.0 1.0;
                  1.0 0.1 1.0 0.1 1.0;
                  0.1 1.0 1.0 1.0 1.0;
                  0.1 0.1 1.0 1.0 0.2])
    kdtree = KDTree(Xf, Euclidean())
    D4 = pairwise(kdtree, Xf, 2; dims=2)
    Dt4 = pairwise(kdtree, transpose(Xf), 2; dims=1)
    @test D4 == Dt4 ≈ Float32.([0.0      Inf        1.55885  Inf        1.27671;
                                1.27279   0.0      Inf       Inf        1.27671;
                                Inf        1.55885   0.0       0.9      Inf;
                                Inf        1.27279  Inf        0.0       1.20416;
                                Inf        1.27671  Inf        1.20416   0.0])

    kdtree = KDTree(Xf, Euclidean())
    D5 = pairwise(kdtree, Xf, 4; dims=2)
    Dt5 = pairwise(kdtree, transpose(Xf), 4; dims=1)
    @test D5 == Dt5 == pairwise(Euclidean(), Xf; dims=2)
end

@testset "Benchmark" begin

    Ts = [Int16, Int32, Int64, Float16, Float32, Float64]
    for T in Ts
        D = SparseDistanceMatrix(1_000_000, Int[], Int[], T[])
        av = T <: Integer ? trunc.(T, randn(9_999)) : T.(randn(9_999))
        [D[1,i+1] = av[i] for i in 1:9_999]

        t = @belapsed $D'
        @test t < 0.001

        @test transpose(D)[1,2] == typemax(T)
        @test transpose(D)[2,1] == av[1]
        @test D'[1,2] == typemax(T)
        @test D'[2,1] == av[1]
    end
end
