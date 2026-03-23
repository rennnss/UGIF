import ee

ee.Initialize(project='astute-baton-486304-r1')
roi = ee.Geometry.Rectangle([80.20, 12.92, 80.25, 12.98])
col = (ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(roi)
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.eq('instrumentMode', 'IW')))

print("Wide Post:", col.filterDate('2023-12-01', '2024-02-01').size().getInfo())
print("Dates:")
def get_date(img):
    return img.date().format('YYYY-MM-dd')
dates = col.filterDate('2023-12-01', '2024-02-01').toList(10).map(get_date).getInfo()
print(dates)
