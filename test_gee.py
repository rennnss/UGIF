import ee
ee.Initialize(project='astute-baton-486304-r1')
roi = ee.Geometry.Rectangle([80.20, 12.92, 80.25, 12.98])
col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(roi).filterDate('2023-12-05', '2023-12-20')
print(f"Total scenes: {col.size().getInfo()}")
col_filtered = col.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
print(f"Scenes < 20% clouds: {col_filtered.size().getInfo()}")
