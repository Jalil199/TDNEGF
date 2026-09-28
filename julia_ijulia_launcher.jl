import IJulia

# Force these methods into the latest world before entering the kernel loop.
Base.pipe_writer(io::IJulia.IJuliaStdio{Base.PipeEndpoint}) = io.io.io
Base.pipe_reader(io::IJulia.IJuliaStdio{Base.PipeEndpoint}) = io.io.io

Base.invokelatest(IJulia.run_kernel)
