using CSV, DataFrames, Plots, Statistics, Colors, Random

function analyze_csv_matrix(csv_file::String, row_split::Int)
    """
    Read CSV file, display as image, and analyze corroded vs bulk regions

    Parameters:
    - csv_file: path to CSV file containing matrix data - row_split: row number that separates corroded (upper) from bulk (lower) regions """

    # Read CSV file into a matrix
    println("Reading CSV file: $csv_file")
    df = CSV.read(csv_file, DataFrame, header=false)
    # Drop the last column if it's entirely missing
    if all(ismissing, df[:, end])
        df = df[:, 1:end-1]
    end
    matrix = Matrix(df)

    # Get matrix dimensions
    n_rows, n_cols = size(matrix)
    println("Matrix dimensions: $n_rows × $n_cols")
    println("Split row: $row_split")

    # Validate row_split parameter
    if row_split < 1 || row_split > n_rows
        error("Row split ($row_split) must be between 1 and $n_rows")
    end

    # Split matrix into regions
    corroded_region = matrix[1:row_split, :]
    bulk_region = matrix[row_split+1:end, :]

    # Calculate statistics for each region
    corroded_mean = mean(corroded_region)
    corroded_std = std(corroded_region)
    corroded_total = sum(corroded_region)
    corroded_pixels = length(corroded_region)
    bulk_mean = mean(bulk_region)
    bulk_std = std(bulk_region)
    bulk_total = sum(bulk_region)
    bulk_pixels = length(bulk_region)

    # Calculate row averages for the entire matrix
    row_averages = [mean(matrix[i, :]) for i in 1:n_rows]

    # Display results
    println("\n--- Region Statistics ---")
    println("Corroded region (rows 1-$row_split):")
    println("  Mean: $(round(corroded_mean, digits=3))")
    println("  Std:  $(round(corroded_std, digits=3))")
    println("  Sum:  $(round(corroded_total, digits=3))")
    println("  Pixels:  $(round(corroded_pixels, digits=1))")
    println("\nBulk region (rows $(row_split+1)-$n_rows):")
    println("  Mean: $(round(bulk_mean, digits=3))")
    println("  Std:  $(round(bulk_std, digits=3))")
    println("  Sum:  $(round(bulk_total, digits=3))")
    println("  Pixels:  $(round(bulk_pixels, digits=1))")

    # Create plots
    p1 = create_image_plot(matrix, row_split)
    p2 = create_row_average_plot(row_averages, row_split)

    # Combine plots
    combined_plot = plot(p1, p2, layout=(1, 2), size=(1000, 400))

    # Display the combined plot
    display(combined_plot)

    # Return results as a named tuple
    return (
        matrix=matrix,
        corroded_stats=(mean=corroded_mean, std=corroded_std, sum=corroded_total, pixels=corroded_pixels),
        bulk_stats=(mean=bulk_mean, std=bulk_std, sumn=bulk_total, pixels=bulk_pixels),
        row_averages=row_averages,
        plot=combined_plot
    )
end

function create_image_plot(matrix, row_split)
    """Create heatmap visualization of the matrix with split line"""

    n_rows, n_cols = size(matrix)

    # Create heatmap
    p = heatmap(matrix,
        c=:grays,
        aspect_ratio=:equal,
        title="Matrix Visualization",
        xlabel="Column",
        ylabel="Row")

    # Add horizontal line to show the split
    hline!([row_split + 0.5],
        color=:red,
        linewidth=3,
        linestyle=:dash,
        label="Corroded/Bulk Split")

    # Add region labels
    annotate!(n_cols * 0.1, row_split * 0.5,
        text("Corroded", :red, :left, 12))
    annotate!(n_cols * 0.1, row_split + (n_rows - row_split) * 0.5,
        text("Bulk", :blue, :left, 12))

    return p
end

function create_row_average_plot(row_averages, row_split)
    """Create plot of row averages with split line"""

    n_rows = length(row_averages)

    # Create line plot of row averages
    p = plot(1:n_rows, row_averages,
        linewidth=2,
        marker=:circle,
        markersize=3,
        title="Row Averages",
        xlabel="Row Number",
        ylabel="Average Value",
        label="Row Average",
        legend=:topright)

    # Add vertical line at the split
    vline!([row_split],
        color=:red,
        linewidth=3,
        linestyle=:dash,
        label="Split at Row $row_split")

    # Color-code the background regions
    plot!(background_color_inside=:white)

    return p
end

function create_sample_csv(filename::String="sample_matrix.csv", rows::Int=20, cols::Int=15)
    """Create a sample CSV file for testing purposes"""

    # Generate sample data with different characteristics for upper and lower regions
    Random.seed!(42)  # For reproducible results

    matrix = zeros(rows, cols)

    # Upper region (corroded) - higher values with more variation
    for i in 1:div(rows, 2)
        for j in 1:cols
            matrix[i, j] = 150 + 50 * randn() + 10 * sin(i * 0.5) * cos(j * 0.3)
        end
    end

    # Lower region (bulk) - lower values with less variation
    for i in (div(rows, 2)+1):rows
        for j in 1:cols
            matrix[i, j] = 80 + 20 * randn() + 5 * sin(i * 0.3) * cos(j * 0.5)
        end
    end

    # Ensure all values are positive
    matrix = max.(matrix, 0)

    # Write to CSV
    CSV.write(filename, DataFrame(matrix, :auto), header=false)
    println("Sample CSV file created: $filename")

    return filename
end

# Example usage:
println("CSV Matrix Analysis Script")
println("="^40)

# Create a sample CSV file for demonstration
sample_file = create_sample_csv("sample_matrix.csv", 25, 20)
sample_file = "../assets/4-point Bending Corrosion/EDS/Purified Salt/Cross Section/x3500 Centerline/Cr Kα1.csv"

# Analyze the matrix with split at row 12
results = analyze_csv_matrix(sample_file, 260)

# You can also use your own CSV file:
# results = analyze_csv_matrix("your_file.csv", 10)
