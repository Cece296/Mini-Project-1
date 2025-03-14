import rasterio
import numpy as np

def calculate_gsd(orthomosaic_path):
    """Extracts Ground Sample Distance (GSD) from an orthomosaic using pixel resolution."""
    with rasterio.open(orthomosaic_path) as dataset:
        # Get pixel resolution in meters per pixel (usually in degrees for geotiffs)
        pixel_size_x, pixel_size_y = dataset.res
        gsd = (pixel_size_x + pixel_size_y) / 2  # Average resolution in meters per pixel
        width, height = dataset.width, dataset.height  # Image dimensions in pixels
        
        # Compute field size in meters
        field_width_m = width * gsd
        field_height_m = height * gsd
        field_area_m2 = field_width_m * field_height_m
        
        return gsd, field_width_m, field_height_m, field_area_m2

def calculate_pumpkin_density(orthomosaic_path, pumpkin_count):
    """Calculates the number of pumpkins per square meter."""
    gsd, field_width_m, field_height_m, field_area_m2 = calculate_gsd(orthomosaic_path)
    
    if field_area_m2 > 0:
        pumpkin_density = pumpkin_count / field_area_m2
    else:
        pumpkin_density = 0  # Avoid division by zero
    
    return gsd, field_width_m, field_height_m, field_area_m2, pumpkin_density


orthomosaic_path = "Gyldensteensvej-9-19-2017-orthophoto (1).tif"  
pumpkin_count = 19711  

gsd, width_m, height_m, area_m2, density = calculate_pumpkin_density(orthomosaic_path, pumpkin_count)

print(f"Ground Sample Distance (GSD): {gsd:.3f} meters per pixel")
print(f"Field Width: {width_m:.2f} meters")
print(f"Field Height: {height_m:.2f} meters")
print(f"Field Area: {area_m2:.2f} square meters")
print(f"Pumpkin Density: {density:.3f} pumpkins per square meter")
