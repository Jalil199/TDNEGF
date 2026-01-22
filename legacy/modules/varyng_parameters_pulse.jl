phases = [pi, 0.0, 3pi/2]
#periods = [600, 300, 150, 100]
pulses = ["smbi.txt", "smle.txt","smre.txt" ]
Jqs = [0.08, 0.1, 0.12]

# Include the 'using' statement for 'contains'
using Base.Iterators: contains

# Loop over different n values
for phase in phases
    for pulse in pulses
        for Jq in Jqs
            # Open the original parameters file
            original_filename = "parameters.txt"
            open(original_filename, "r") do file
                # Create a new file for the modified parameters
                round_phase = round(phase, digits=2)
                modified_filename = "P$(round_phase)_Pol$(pulse)_Jq$(Jq).txt"
                new_file = open(modified_filename, "w")
		#println("""bias_file = "./vtd.txt" """)
                # Loop through lines in the original file
                for line in eachline(file)
                    if contains(line, "θ = ")
                        modified_line = "θ = $(phase)"
                    elseif contains(line, "J_qsl = ")
                        modified_line = "J_qsl = $(Jq) "
                    elseif contains(line, "bias_file = vtd")
                        modified_line = "bias_file = $(pulse)"
                    elseif contains(line, "name = ")
                        modified_line = "name = $(modified_filename)"
                    else
                        modified_line = line
                    end

                    println(new_file, modified_line)
                end

                # Close the new file
                close(new_file)
            end
        end
    end
end

