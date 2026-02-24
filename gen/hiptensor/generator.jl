using Clang.Generators
using JuliaFormatter

include_dir = normpath(joinpath(ENV["ROCM_PATH"], "include"))
hiptensor_dir = joinpath(include_dir, "hiptensor")
options = load_options("hiptensor/hiptensor-generator.toml")

args = get_default_args()
push!(args, "-I$include_dir")

# hipTensor headers are C++ (.hpp) with STL includes that Clang.jl's C mode
# can't parse. Extract only the C-compatible API surface into a plain C header.
types_hpp = read(joinpath(hiptensor_dir, "hiptensor_types.hpp"), String)
api_hpp = read(joinpath(hiptensor_dir, "hiptensor.hpp"), String)

open("./hiptensor.h", "w") do io
    println(io, """
    #include <stddef.h>
    #include <stdint.h>

    typedef void* hipStream_t;
    typedef struct FILE FILE;
    """)

    # Extract enum and typedef blocks from types header.
    for line in split(types_hpp, '\n')
        stripped = strip(line)
        startswith(stripped, "#include <") && continue
        startswith(stripped, "#include \"hip/") && continue
        startswith(stripped, "#include \"internal/") && continue
        startswith(stripped, "#ifndef") && continue
        startswith(stripped, "#define HIPTENSOR_TYPES") && continue
        startswith(stripped, "#endif") && continue
        println(io, line)
    end

    println(io)

    # Extract function declarations from API header.
    for line in split(api_hpp, '\n')
        stripped = strip(line)
        startswith(stripped, "#") && continue
        startswith(stripped, "//") && continue
        println(io, line)
    end
end

headers = ["./hiptensor.h"]
ctx = create_context(headers, args, options)

build!(ctx, BUILDSTAGE_NO_PRINTING)

function rewrite!(e::Expr)
    if e.head === :const
        @assert Meta.isexpr(e.args[1], :(=))
        rhs = e.args[1].args[2]
        if Meta.isexpr(rhs, :call)
            if rhs.args[1] == :(*) && rhs.args[3] == :f
                e.args[1].args[2] = :(Float32($(rhs.args[2])))
            elseif rhs.args[1] == :(Cuint)
                e.args[1].args[2] = :($(rhs.args[2]) % Cuint)
            end
        end
        return e
    end
    (e.head === :function && Meta.isexpr(e.args[1], :call)) || return e
    f = e.args[1].args[1]
    if !(f isa Symbol)
        @assert f in (:(Base.getproperty), :(Base.setproperty!), :(Base.propertynames))
        return e
    end
    stmts = e.args[2].args
    map!(stmts, stmts) do ex
        Meta.isexpr(ex, :macrocall) || return ex
        ex.args[1] === Symbol("@ccall") || return ex
        Expr(:macrocall, Symbol("@check"), nothing, ex)
    end
    pushfirst!(stmts, :(AMDGPU.prepare_state()))
    return e
end

function rewrite!(dag::ExprDAG)
    for node in get_nodes(dag)
        for expr in get_exprs(node)
            rewrite!(expr)
        end
    end
end

rewrite!(ctx.dag)

build!(ctx, BUILDSTAGE_PRINTING_ONLY)

path = options["general"]["output_file_path"]
format_file(path, YASStyle())
