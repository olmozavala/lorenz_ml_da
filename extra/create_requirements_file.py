# %%
import os
import pkg_resources

# Step 1: Collect all unique imports from your project files
project_imports = set()
for subdir, dirs, files in os.walk('.'):
	for file in files:
		if file.endswith('.py'):
			with open(os.path.join(subdir, file), 'r') as f:
				for line in f:
					if line.startswith('import ') or line.startswith('from '):
						imported_module = line.split()[1].split('.')[0]
						project_imports.add(imported_module)

# Step 2: Filter out standard library imports and identify package names
external_packages = set()
for imp in project_imports:
	try:
		# This will fail if 'imp' is part of the standard library
		package = pkg_resources.get_distribution(imp).project_name
		external_packages.add(package)
	except pkg_resources.DistributionNotFound:
		pass

# %% Now 'external_packages' contains the names of packages that you likely need to include in your requirements.txt
# Save into requirements.txt
with open('requirements.txt', 'w') as f:
    for package in external_packages:
        f.write(f'{package}\n')
        
# %%
