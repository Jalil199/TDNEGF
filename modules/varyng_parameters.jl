phases = [pi, 0.0, 3pi/2]
periods = [600, 300, 150, 100]
Jqs = [0.02,0.04,0.06,0.08, 0.1, 0.12,0.14,0.16,0.18,0.2]

# Include the 'using' statement for 'contains'
using Base.Iterators: contains

# Loop over different n values
for phase in phases
    for period in periods
        for Jq in Jqs
            # Open the original parameters file
            original_filename = "parameters.txt"
            open(original_filename, "r") do file
                # Create a new file for the modified parameters
                round_phase = round(phase, digits=2)
                modified_filename = "P$(round_phase)_T$(period)_Jq$(Jq).txt"
                new_file = open(modified_filename, "w")

                # Loop through lines in the original file
                for line in eachline(file)
                    if contains(line, "θ = ")
                        modified_line = "θ = $(phase)"
                    elseif contains(line, "J_qsl = ")
                        modified_line = "J_qsl = $(Jq) "
                    elseif contains(line, "period = ")
                        modified_line = "period = $(period)"
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

