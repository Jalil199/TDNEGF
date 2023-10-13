module read_parameters 
### En este modulo se definen todas las funciones relaciones con la lectura de los parametros 
##include("parameters.jl")
##using .parameters
using DelimitedFiles         ### Manipulate files 
# Nombre del archivo de parámetros
archivo_parametros = "./modules/parameters.txt"
# Diccionario para almacenar los parámetros
# Abrir y leer el archivo de parámetros
function read_params(;archivo_parametros=archivo_parametros)
    parametros = Dict{String, Any}()
    open(archivo_parametros) do file
        for linea in eachline(file)
            # Check if the line is a comment (starts with ##) and skip it
            # Split the line by '##' and take only the part before '##'
            parts = split(linea, "#")
            cleaned_line = strip(parts[1])
            isempty(cleaned_line) && continue
            # Dividir la línea en nombre y valor utilizando una coma como separador si está presente
            # Si no hay comas, proceder como antes
            # Dividir la línea en nombre y valor
            parts = split(cleaned_line, "=")
            if length(parts) == 2
                nombre = strip(parts[1])
                valor = strip(parts[2])
             # Intentar convertir el valor a un tipo numérico o cadena
               if occursin(",", valor)
                    valores = split(valor, ",")
                    #println("valores", valores[1])
                    # Intentar convertir los elementos en números de punto flotante
                    try
                        valores = Int64[parse(Int64, v) for v in valores]
                    catch
                        try 
                            ### try to conver into a an array of Float pointers
                            valores = Float64[parse(Float64, v) for v in valores]
                        catch
                            # Si no se pueden convertir, se mantienen como cadena
                        end
                        
                    end
                    # Almacenar el parámetro en el diccionario
                    parametros[nombre] = valores
                else
                    try
                        # Si sabes que el valor es un entero, usa parse(Int, valor)
                        valor = parse(Int, valor)
                    catch
                        try
                            # Si sabes que el valor es un número de punto flotante, usa parse(Float64, valor)
                            valor = parse(Float64, valor)
                        catch
                            try 
                                valor = parse(Bool, valor)
                            catch
                                valor = strip(valor, '"')  # Eliminar comillas si es una cadena
                            end
                        end
                    end
                    # Almacenar el parámetro en el diccionario
                    parametros[nombre] = valor
                end
             
            end
        end
    end
    ##println(parametros)
    parametros
end


#println(typeof(read_params()["read_bias_file"]) )
end

