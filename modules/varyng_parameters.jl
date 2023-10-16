# Number of different n values to generate
num_values = 2

# Loop over different n values
for i in 1:num_values
    n_value = i * 2  # You can adjust this calculation as needed

    # Open the original parameters file
    original_filename = "parameters.txt"
    open(original_filename, "r") do file
        # Create a new file for the modified parameters
        modified_filename = "parameters_n_$n_value.txt"
        new_file = open(modified_filename, "w")

        # Loop through lines in the original file
        for line in eachline(file)
            # Check if the line contains "n ="
            if contains(line, "n =")
                # Modify the line with the new n value
                modified_line = "n = $n_value"
            else
                # Keep the line as-is
                modified_line = line
            end

            # Write the modified line to the new file
            println(new_file, modified_line)
        end

        # Close the new file
        close(new_file)
    end
end
