using NPZ
include("Lattice2D.jl")


data_path = "../../data2"
if !isdir(data_path)
	mkpath(data_path)
end


matrix_to_tensor(matrix) = reshape(convert(Array{Int8}, matrix), size(matrix)..., 1)


function generate_data(
	L,
	n_samples,
	p_occupations;
	extra_sample_range=(0.55,0.65),
	extra_sample_rate=10,
	as_dict=false
	)

	N_samples = n_samples * length(p_occupations)
	extra = 0
	if extra_sample_rate > 1
		extra = Int(
			sum(extra_sample_range[1] .≤ p_occupations .≤ extra_sample_range[2]) * (extra_sample_rate - 1)
			)
	end
	N_samples += extra * n_samples

	X = zeros(Int8, L, L, 1, N_samples)
	Y = zeros(Int8, N_samples)
	p = zeros(N_samples)

	i = 1
	for p_occupation in p_occupations
		if 0.55 ≤ p_occupation ≤ 0.65
			n_samples_effective = n_samples * extra_sample_rate
		else
			n_samples_effective = n_samples
		end

		for sample in 1:n_samples_effective
			lattice = Lattice2D(L, p_occupation, i)
			X[:, :, :, i] = matrix_to_tensor(lattice.occupied)
			Y[i] = lattice.is_percolating
			p[i] = p_occupation

			i += 1
		end
	end

	if as_dict
		return Dict("X" => X, "Y" => Y, "p" => p)
	else
		return (X = X, Y = Y, p = p)
	end
end


# %% train and test data generation


L = 32
n_samples = 150
p_occupations = [
	range(0, stop=0.54, length=55);
	range(0.55, stop=0.65, length=51);
	range(0.66, stop=1, length=35)
	];

@time data_train_test  = generate_data(L, n_samples, p_occupations, as_dict=true);
size(data_train_test["X"])
npzwrite(joinpath(data_path, "data_train_test.npz"), data_train_test)


# %% transition analysis data generation


p_c = 0.5927
L = 32
n_samples = 5000
p_occupations = collect(range(0.5, stop=0.65, step=1e-2));

@time data_transition = generate_data(L, n_samples, p_occupations, extra_sample_rate=1, as_dict=true);
size(data_transition["X"])
npzwrite(joinpath(data_path, "data_transition.npz", data_transition)
