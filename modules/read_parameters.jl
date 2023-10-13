module read_parameters 
### En este modulo se definen todas las funciones relaciones con la lectura de los parametros 
##include("parameters.jl")
##using .parameters
using DelimitedFiles         ### Manipulate files 
# Nombre del archivo de parámetros
archivo_parametros = "parameters.txt"
# Diccionario para almacenar los parámetros
parametros = Dict{String, Any}()
# Abrir y leer el archivo de parámetros
function read_params(parametros=parametros, archivo_parametros=archivo_parametros)
    open(archivo_parametros) do file
        for linea in eachline(file)
            # Dividir la línea en nombre y valor utilizando una coma como separador si está presente
            # Si no hay comas, proceder como antes
            # Dividir la línea en nombre y valor
            parts = split(linea, "=")
            if length(parts) == 2
                nombre = strip(parts[1])
                valor = strip(parts[2])
             # Intentar convertir el valor a un tipo numérico o cadena
               if occursin(",", valor)
                    valores = split(valor, ",")
                    # Intentar convertir los elementos en números de punto flotante
                    try
                        valores = Float64[parse(Float64, v) for v in valores]
                    catch
                        # Si no se pueden convertir, se mantienen como cadena
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
                            valor = strip(valor, '"')  # Eliminar comillas si es una cadena
                        end
                    end
                # Almacenar el parámetro en el diccionario
                parametros[nombre] = valor
                end
             
            end
        end
    end


        
end


