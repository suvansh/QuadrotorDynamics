QuadrotorDynamics


data_dir = "/Users/sanjeev/Downloads/processed_data/"

function max_accel(filepath)
    xs, ẋs, us, dt = QuadrotorDynamics.read_neuroBEM(filepath)
    norm(ẋs[:,8:10])
    norm(ẋs[:,11:13])
end

max_accels = Dict()
for file in readdir(data_dir)
    filepath = joinpath(path, file)
    max_accels[file] = max_accel(filepath)
end
