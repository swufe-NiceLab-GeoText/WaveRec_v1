import numpy as np
# Load the X.npy file
X_data = np.load('../TaxiBJ/P1/test/ext.npy')

# Get some basic information about the data
X_data_shape = X_data.shape
X_data_dtype = X_data.dtype
X_sample_data = X_data[:5]  # Displaying the first 5 entries for a quick view

print(X_data_shape, X_data_dtype, X_sample_data)