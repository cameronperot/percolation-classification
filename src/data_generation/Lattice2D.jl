using Random


mutable struct Lattice2D
	L              ::Int
	p_occupation   ::Float64
	occupied       ::BitArray{2}
	cluster_ids    ::Matrix{Int}
	equivalent_ids ::Vector{Int}
	clusters       ::Dict{Int, Set{NTuple{2, Int}}}
	is_percolating ::Bool
	rng            ::AbstractRNG

	function Lattice2D(L::Int, p_occupation::Float64, seed::Int)
		rng            = MersenneTwister(seed)
		occupied       = rand(rng, L, L) .< p_occupation
		cluster_ids    = zeros(Int, size(occupied))
		clusters       = Dict{Int, Set{NTuple{2, Int}}}()
		equivalent_ids = collect(1:L^2)
		is_percolating = false

		lattice = new(
			L,
			p_occupation,
			occupied,
			cluster_ids,
			equivalent_ids,
			clusters,
			is_percolating,
			rng
		)

		label_clusters!(lattice)
		create_cluster_sets!(lattice)
		is_lattice_percolating!(lattice)

		return lattice
	end
end


function uf_find(lattice::Lattice2D, cluster_id::Int)
	while lattice.equivalent_ids[cluster_id] ≠ cluster_id
		cluster_id = lattice.equivalent_ids[cluster_id]
	end
	return cluster_id
end


function uf_union!(lattice::Lattice2D, cluster_id_1::Int, cluster_id_2::Int)
	lattice.equivalent_ids[uf_find(lattice, cluster_id_1)] = uf_find(lattice, cluster_id_2)
	return uf_find(lattice, cluster_id_1)
end


function label_clusters!(lattice::Lattice2D)
	new_cluster_id = 1
	for j in 1:lattice.L, i in 1:lattice.L
		if lattice.occupied[i, j]
			left_cluster_id = (j == 1 ? 0 : lattice.cluster_ids[i, j - 1])
			up_cluster_id   = (i == 1 ? 0 : lattice.cluster_ids[i - 1, j])

			if left_cluster_id == up_cluster_id == 0
				lattice.cluster_ids[i, j] = new_cluster_id
				new_cluster_id += 1
			elseif left_cluster_id == 0 || up_cluster_id == 0
				lattice.cluster_ids[i, j] = maximum((left_cluster_id, up_cluster_id))
			else
				lattice.cluster_ids[i, j] = uf_union!(lattice, left_cluster_id, up_cluster_id)
			end
		end
	end

	for j in 1:lattice.L, i in 1:lattice.L
		if lattice.occupied[i, j]
			lattice.cluster_ids[i, j] = uf_find(lattice, lattice.cluster_ids[i, j])
		end
	end
end


function create_cluster_sets!(lattice::Lattice2D)
	clusters    = Dict{Int, Set{NTuple{2, Int}}}()
	for j in 1:lattice.L, i in 1:lattice.L
		if lattice.occupied[i, j]
			site = (i, j)
			if haskey(clusters, lattice.cluster_ids[i, j])
				union!(clusters[lattice.cluster_ids[i, j]], [site])
			else
				clusters[lattice.cluster_ids[i, j]] = Set([site])
			end
		end
	end

	lattice.clusters = clusters
end


function is_lattice_percolating!(lattice::Lattice2D)
	L = lattice.L
	top_boundary    = Set{NTuple{2, Int}}([(1, j) for j in 1:L])
	bottom_boundary = Set{NTuple{2, Int}}([(L, j) for j in 1:L])
	left_boundary   = Set{NTuple{2, Int}}([(i, 1) for i in 1:L])
	right_boundary  = Set{NTuple{2, Int}}([(i, L) for i in 1:L])

	for cluster in values(lattice.clusters)
		if length(cluster ∩ top_boundary) > 0 && length(cluster ∩ bottom_boundary) > 0
			lattice.is_percolating = true
			break
		elseif length(cluster ∩ left_boundary) > 0 && length(cluster ∩ right_boundary) > 0
			lattice.is_percolating = true
			break
		end
	end
end
