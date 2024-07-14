import pandas as pd
import numpy as np

def add_time_step(file_name, output_file_name):
    # Load the CSV file. Assuming the file is properly comma-separated.
    df = pd.read_csv(file_name)
    
    # Add a new column 'time_step' with random integers from 1 to 7
    np.random.seed(42)  # Optional: for reproducible results
    df['time_step'] = np.random.randint(1, 8, size=len(df))
    
    # Save the modified DataFrame back to a new CSV file
    df.to_csv(output_file_name, index=False)
    
    # Print the head of the updated DataFrame to check the new column
    print(df.head())

# Apply the function to both files
add_time_step('./cross_validation/train_CV0_size1000.csv', './cross_validation/train_CV0_size1000_updated.csv')
add_time_step('./cross_validation/test_CV0.csv', './cross_validation/test_CV0_updated.csv')
