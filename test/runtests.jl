using SparseDistanceMatrices, BenchmarkTools

using Test

@testset "SparseDistanceMatrices" begin

    N = 100
    Ts = [Int16, Int32, Int64, Float16, Float32, Float64]

    for T in Ts
        D = SparseDistanceMatrix(N, Int[], Int[], T[])
        a = T <: Integer ? trunc(T, 10.0) : T(10.0)
        b = T <: Integer ? trunc(T, 12.3) : T(12.3)
        D[1,20] = a
        D[40,2] = b

        @test size(D) == (100, 100)
        [@test D[i,i] == T(0.0) for i in 1:N]
        @test D[1,100] == typemax(T)
        @test D[1,20] == a
        @test D[40,2] == b
        @test sum(D .!== typemax(T)) == 2+N
        @test D != D'
        @test symmetrize!(D) == D'

        ndval = T <: Integer ? trunc(T, 1000.0) : T(1000.0)
        D = SparseDistanceMatrix(N, Int[], Int[], T[], ndval)
        a = T <: Integer ? trunc(T, 10.0) : T(10.0)
        b = T <: Integer ? trunc(T, 12.3) : T(12.3)
        D[1,20] = a
        D[40,2] = b

        @test size(D) == (100, 100)
        [@test D[i,i] == T(0.0) for i in 1:N]
        @test D[1,100] == ndval
        @test D[1,20] == a
        @test D[40,2] == b
        @test sum(D .!== ndval) == 2+N
        @test D != D'
        @test symmetrize!(D) == D'

        # check double entry
        D[40,2] = a
        @test D[40,2] == a
        @test length(D.rowindices) == length(D.colindices) == length(D.ndval) == 4

        @test_throws ArgumentError D[1,1] = T <: Integer ? trunc(T, 10.0) : T(10.0)
    end

end

@testset "Benchmark" begin

    Ts = [Int16, Int32, Int64, Float16, Float32, Float64]
    for T in Ts
        D = SparseDistanceMatrix(1_000_000, Int[], Int[], T[])
        a = T <: Integer ? trunc(T, 10.0) : T(10.0)
        D[1,2] = a

        t = @belapsed $D'
        @test t < 0.001

        @test D'[1,2] == typemax(T)
        @test D'[2,1] == a
    end
end
